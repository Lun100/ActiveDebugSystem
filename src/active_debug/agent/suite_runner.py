"""Four-strategy active diagnosis, localization, and repair loop for v3 suite cases."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from time import monotonic
from typing import Any, Protocol

from active_debug.agent.agent import Actions, BudgetExhausted, WallClockExhausted
from active_debug.agent.debug_controller import REPAIR_ACTION_RESERVE
from active_debug.agent.llm_client import (
    CallUsage,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderResponseError,
    _extract_json_object,
    _extract_python_source,
)
from active_debug.agent.suite_executor import SuiteCase, SuiteToolExecutor
from active_debug.agent.trace_logger import TraceLogger
from active_debug.diagnosis import (
    belief_updater,
    experiment_selector,
    hypothesis_manager,
    stopping_policy,
)
from active_debug.schemas import (
    DiagnosisState,
    Experiment,
    Hypothesis,
    Observation,
    Patch,
    RunConfig,
    Strategy,
)


@dataclass(frozen=True)
class HypothesisSpec:
    """Candidate fault cause paired with optional Class.method localization."""

    name: str
    description: str
    prior: float
    target_function: str | None = None


@dataclass(frozen=True)
class ProbeSpec:
    """Self-contained Python snippet and predicted stdout under each hypothesis."""

    code: str
    predictions: dict[str, str]
    cost: float = 1.0


@dataclass(frozen=True)
class SuiteDiagnosisPlan:
    """LLM-synthesized hypothesis space and candidate discriminating probes."""

    hypotheses: tuple[HypothesisSpec, ...]
    probes: tuple[ProbeSpec, ...]


@dataclass(frozen=True)
class SuiteFreeDecision:
    """Next step chosen by free_debug strategy."""

    action: str  # "probe" | "repair"
    code: str | None = None
    predicted_function: str | None = None


@dataclass(frozen=True)
class SuiteRepairProposal:
    """Candidate full-module patch and optional pre-repair localization."""

    patch: Patch
    predicted_function: str | None = None


@dataclass(frozen=True)
class SuiteRunOutcome:
    """Result of executing one (case, strategy, seed) run on a v3 bundle case."""

    case_id: str
    revision: str
    seed: int
    strategy: Strategy
    status: str  # "completed" | "timeout" | "error"
    stop_reason: str
    actions: int
    seconds: float
    final_source: str
    source_modified: bool
    predicted_function: str | None = None
    error_detail: str | None = None


class SuiteProvider(Protocol):
    last_usage: CallUsage | None

    def hypothesize(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        seed: int,
    ) -> SuiteDiagnosisPlan: ...

    def choose_probe(
        self,
        case: SuiteCase,
        state: DiagnosisState,
        plan: SuiteDiagnosisPlan,
        options: list[Experiment],
        probe_codes: dict[int, str],
        seed: int,
    ) -> Experiment: ...

    def free_probe(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        feedback: str,
        seed: int,
    ) -> SuiteFreeDecision: ...

    def repair(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        diagnosis_summary: str,
        feedback: str,
        seed: int,
    ) -> SuiteRepairProposal: ...


def normalize_predicted_function(
    raw_name: str | None, callable_symbols: tuple[str, ...]
) -> str | None:
    """Canonicalize an LLM function prediction against AST symbols (e.g. Class.method)."""
    if not raw_name:
        return None
    cleaned = raw_name.strip().strip("`'\"").split("(")[0].strip()
    if not cleaned:
        return None
    if not callable_symbols:
        return cleaned
    if cleaned in callable_symbols:
        return cleaned
    suffix_matches = [s for s in callable_symbols if s.endswith(f".{cleaned}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    # Also accept bare method if LLM wrote buggy.func or similar prefix
    short = cleaned.split(".")[-1]
    if short in callable_symbols:
        return short
    short_matches = [s for s in callable_symbols if s.endswith(f".{short}")]
    if len(short_matches) == 1:
        return short_matches[0]
    return cleaned


def _normalize_priors(raw_weights: list[float]) -> list[float]:
    cleaned = [w if (isfinite(w) and w > 0) else 0.0 for w in raw_weights]
    total = sum(cleaned)
    n = len(raw_weights)
    if total <= 0:
        base = 1.0 / n
        priors = [base] * n
    else:
        priors = [w / total for w in cleaned]
    # Adjust last entry to absorb IEEE-754 rounding so sum == 1.0 strictly
    priors[-1] = max(0.0, 1.0 - sum(priors[:-1]))
    return priors


def _format_obs(obs: Observation) -> str:
    return (
        f"returncode={obs.returncode}, timed_out={obs.timed_out}\n"
        f"stdout:\n{obs.stdout.strip() or '(empty)'}\n"
        f"stderr:\n{obs.stderr.strip() or '(empty)'}"
    )


def _format_suite_diagnosis_summary(
    state: DiagnosisState,
    plan: SuiteDiagnosisPlan,
    probe_codes: dict[int, str],
) -> str:
    spec_by_name = {h.name: h for h in plan.hypotheses}
    ranked = sorted(state.hypotheses, key=lambda h: h.belief, reverse=True)
    lines = ["Active diagnosis posterior beliefs:"]
    for h in ranked:
        spec = spec_by_name.get(h.name)
        fn_tag = f" [function={spec.target_function}]" if spec and spec.target_function else ""
        desc = f": {spec.description}" if spec else ""
        lines.append(f"  - {h.name} (belief={h.belief:.4f}){fn_tag}{desc}")
    if state.history:
        lines.append("Executed diagnostic probes:")
        for idx, (exp, obs) in enumerate(state.history, start=1):
            snippet = probe_codes.get(exp.argument, "").strip()
            lines.append(
                f"  Probe #{idx} (cost={exp.cost}):\n"
                f"    code: {snippet!r}\n"
                f"    predictions: {exp.predictions}\n"
                f"    actual stdout={obs.stdout.strip()!r}, returncode={obs.returncode}, "
                f"timed_out={obs.timed_out}"
            )
    return "\n".join(lines)


class OpenAISuiteAdapter:
    """Drive v3 suite diagnosis, probing, and repair using OpenAICompatibleProvider."""

    def __init__(self, client: OpenAICompatibleProvider):
        self.client = client
        self.last_usage: CallUsage | None = None

    def _call_chat(self, system_prompt: str, user_prompt: str, seed: int) -> str:
        self.last_usage = None
        out = self.client.chat(system_prompt, user_prompt, seed=seed)
        self.last_usage = self.client.last_usage
        return out

    def hypothesize(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        seed: int,
    ) -> SuiteDiagnosisPlan:
        symbols_str = ", ".join(case.callable_symbols) or "(top-level module)"
        system_prompt = (
            "You are an active fault-diagnosis engine for Python debugging. "
            "Given buggy.py and a failing public pytest test, propose 2 to 4 mutually exclusive, "
            "falsifiable root-cause hypotheses and 2 to 4 small self-contained Python probe scripts "
            "that distinguish between those hypotheses.\n"
            "Rules for probes:\n"
            "1. Each probe runs as a standalone script with `buggy` importable on PYTHONPATH.\n"
            "2. Catch expected exceptions inside the probe and print a deterministic single-line "
            "token (e.g. `print(repr(res))` or `print('ERR:' + type(e).__name__)`) so returncode is 0.\n"
            "3. Provide the exact single-line stdout string predicted by EACH hypothesis.\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "hypotheses": [\n'
            '    {"id": "H1", "function": "Class.method_or_func", "description": "...", "prior": 0.5},\n'
            '    {"id": "H2", "function": "Class.method_or_func", "description": "...", "prior": 0.5}\n'
            "  ],\n"
            '  "probes": [\n'
            '    {"code": "import buggy\\n...", "cost": 1.0, "predictions": {"H1": "...", "H2": "..."}}\n'
            "  ]\n"
            "}"
        )
        user_prompt = (
            f"Case ID: {case.case_id} ({case.topic})\n"
            f"Callable symbols in buggy.py: {symbols_str}\n\n"
            f"buggy.py:\n```python\n{source}\n```\n\n"
            f"Public test (test_public.py):\n```python\n{case.public_test_source}\n```\n\n"
            f"Initial public test failure:\n{_format_obs(failure)}"
        )
        raw = self._call_chat(system_prompt, user_prompt, seed)
        obj = _extract_json_object(raw)

        raw_hyps = obj.get("hypotheses")
        if not isinstance(raw_hyps, list) or not raw_hyps:
            raise ProviderResponseError("hypothesize response must contain non-empty 'hypotheses'")

        seen_ids: set[str] = set()
        parsed_hyps: list[tuple[str, str, float, str | None]] = []
        for idx, item in enumerate(raw_hyps[:6], start=1):
            if not isinstance(item, dict):
                continue
            hid = str(item.get("id") or item.get("name") or f"H{idx}").strip()
            if not hid or hid in seen_ids:
                hid = f"H{idx}"
            seen_ids.add(hid)
            desc = str(item.get("description") or hid).strip()
            try:
                prior_val = float(item.get("prior", 1.0))
            except (TypeError, ValueError):
                prior_val = 1.0
            fn_name = normalize_predicted_function(
                str(item["function"]) if item.get("function") is not None else None,
                case.callable_symbols,
            )
            parsed_hyps.append((hid, desc, prior_val, fn_name))

        if not parsed_hyps:
            raise ProviderResponseError("no valid hypotheses parsed from response")

        priors = _normalize_priors([h[2] for h in parsed_hyps])
        hyp_specs = tuple(
            HypothesisSpec(name=h[0], description=h[1], prior=p, target_function=h[3])
            for h, p in zip(parsed_hyps, priors)
        )
        hyp_names = [h.name for h in hyp_specs]

        raw_probes = obj.get("probes")
        probe_specs: list[ProbeSpec] = []
        if isinstance(raw_probes, list):
            for p_item in raw_probes[:6]:
                if not isinstance(p_item, dict):
                    continue
                code = str(p_item.get("code") or "").strip()
                if not code:
                    continue
                raw_preds = p_item.get("predictions")
                pred_map = raw_preds if isinstance(raw_preds, dict) else {}
                complete_preds = {
                    h_name: str(pred_map.get(h_name, "")).strip() for h_name in hyp_names
                }
                try:
                    cost_val = float(p_item.get("cost", 1.0))
                except (TypeError, ValueError):
                    cost_val = 1.0
                if not isfinite(cost_val) or cost_val <= 0:
                    cost_val = 1.0
                probe_specs.append(
                    ProbeSpec(code=code, predictions=complete_preds, cost=cost_val)
                )

        return SuiteDiagnosisPlan(hypotheses=hyp_specs, probes=tuple(probe_specs))

    def choose_probe(
        self,
        case: SuiteCase,
        state: DiagnosisState,
        plan: SuiteDiagnosisPlan,
        options: list[Experiment],
        probe_codes: dict[int, str],
        seed: int,
    ) -> Experiment:
        system_prompt = (
            "You are selecting the next diagnostic probe for Python fault localization. "
            "Respond ONLY with JSON {\"probe_id\": <int>} matching one of the candidate options."
        )
        opt_lines = [
            f"- probe_id={opt.argument}, cost={opt.cost}, predictions={opt.predictions}\n"
            f"  code:\n```python\n{probe_codes.get(opt.argument, '')}\n```"
            for opt in options
        ]
        user_prompt = (
            f"Case: {case.case_id}\n"
            f"{_format_suite_diagnosis_summary(state, plan, probe_codes)}\n\n"
            f"Candidate probes:\n" + "\n".join(opt_lines)
        )
        raw = self._call_chat(system_prompt, user_prompt, seed)
        obj = _extract_json_object(raw)
        chosen_id = obj.get("probe_id", obj.get("argument"))
        if type(chosen_id) is int:
            for opt in options:
                if opt.argument == chosen_id:
                    return opt
        if type(obj.get("index")) is int:
            idx = obj["index"]
            if 0 <= idx < len(options):
                return options[idx]
        raise ProviderResponseError(f"LLM selected invalid probe option: {obj!r}")

    def free_probe(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        feedback: str,
        seed: int,
    ) -> SuiteFreeDecision:
        symbols_str = ", ".join(case.callable_symbols) or "(top-level module)"
        system_prompt = (
            "You are debugging buggy.py against a failing test_public.py. "
            "Either run one standalone Python probe script (importing `buggy` and printing "
            "observations to stdout) OR proceed to repair.\n"
            "Respond ONLY with JSON:\n"
            '{"action": "probe", "code": "import buggy\\nprint(...)"}\n'
            "OR\n"
            '{"action": "repair", "predicted_function": "Class.method_or_func"}'
        )
        user_prompt = (
            f"Case ID: {case.case_id} ({case.topic})\n"
            f"Callable symbols: {symbols_str}\n\n"
            f"Current buggy.py:\n```python\n{source}\n```\n\n"
            f"Public test (test_public.py):\n```python\n{case.public_test_source}\n```\n\n"
            f"Public test failure:\n{_format_obs(failure)}\n\n"
            f"Previous probe/verify feedback:\n{feedback or '(none)'}"
        )
        raw = self._call_chat(system_prompt, user_prompt, seed)
        obj = _extract_json_object(raw)
        action = str(obj.get("action", "")).strip().lower()
        pred_fn = normalize_predicted_function(
            str(obj["predicted_function"]) if obj.get("predicted_function") is not None else None,
            case.callable_symbols,
        )
        if action == "repair":
            return SuiteFreeDecision(action="repair", predicted_function=pred_fn)
        if action == "probe":
            code = str(obj.get("code") or "").strip()
            if not code:
                raise ProviderResponseError("free_probe 'probe' action requires non-empty 'code'")
            return SuiteFreeDecision(action="probe", code=code, predicted_function=pred_fn)
        raise ProviderResponseError(f"unsupported free_probe action: {action!r}")

    def repair(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        diagnosis_summary: str,
        feedback: str,
        seed: int,
    ) -> SuiteRepairProposal:
        symbols_str = ", ".join(case.callable_symbols) or "(top-level module)"
        system_prompt = (
            "You are an expert Python program repair agent. "
            "Fix the bug in buggy.py so both the public test and general edge cases pass, "
            "without overfitting to literal test inputs and while preserving all public APIs.\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "predicted_function": "Class.method_or_func",\n'
            '  "source": "<complete replacement Python source for buggy.py>"\n'
            "}\n"
            "(If escaping code in JSON is inconvenient, you may instead output a single "
            "```python ... ``` block containing the full replacement file.)"
        )
        sections = [
            f"Case ID: {case.case_id} ({case.topic})",
            f"Callable symbols in initial buggy.py: {symbols_str}",
            f"Current buggy.py:\n```python\n{source}\n```",
            f"Public test (test_public.py):\n```python\n{case.public_test_source}\n```",
            f"Latest public test failure:\n{_format_obs(failure)}",
        ]
        if diagnosis_summary:
            sections.append(diagnosis_summary)
        if feedback:
            sections.append(f"Accumulated feedback:\n{feedback}")

        raw = self._call_chat(system_prompt, "\n\n".join(sections), seed)
        pred_fn: str | None = None
        try:
            obj = _extract_json_object(raw)
            if obj.get("predicted_function") is not None:
                pred_fn = normalize_predicted_function(
                    str(obj["predicted_function"]), case.callable_symbols
                )
        except ProviderResponseError:
            pass
        extracted = _extract_python_source(raw)
        try:
            patch = Patch(extracted)
        except ValueError as exc:
            raise ProviderResponseError(f"invalid repair patch: {exc}") from exc
        return SuiteRepairProposal(patch=patch, predicted_function=pred_fn)


class MockSuiteAdapter:
    """Deterministic offline suite provider for unit/integration testing and dry runs."""

    def __init__(self, repair_rules: dict[str, tuple[str, str]] | None = None):
        self.last_usage: CallUsage | None = None
        self.repair_rules = dict(repair_rules or {"demo/1": ("return a - b", "return a + b")})

    def hypothesize(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        seed: int,
    ) -> SuiteDiagnosisPlan:
        self.last_usage = CallUsage()
        fn1 = case.callable_symbols[0] if case.callable_symbols else "buggy"
        fn2 = case.callable_symbols[-1] if len(case.callable_symbols) > 1 else fn1
        hyps = (
            HypothesisSpec("H_primary", f"defect in {fn1}", 0.5, fn1),
            HypothesisSpec("H_secondary", f"defect in {fn2}", 0.5, fn2),
        )
        probes = (
            ProbeSpec(
                code="import buggy\nprint('same')",
                predictions={"H_primary": "same", "H_secondary": "same"},
                cost=1.0,
            ),
            ProbeSpec(
                code="import buggy\nprint('H1_signal')",
                predictions={"H_primary": "H1_signal", "H_secondary": "H2_signal"},
                cost=1.0,
            ),
        )
        return SuiteDiagnosisPlan(hypotheses=hyps, probes=probes)

    def choose_probe(
        self,
        case: SuiteCase,
        state: DiagnosisState,
        plan: SuiteDiagnosisPlan,
        options: list[Experiment],
        probe_codes: dict[int, str],
        seed: int,
    ) -> Experiment:
        self.last_usage = CallUsage()
        return options[-1]

    def free_probe(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        feedback: str,
        seed: int,
    ) -> SuiteFreeDecision:
        self.last_usage = CallUsage()
        fn = case.callable_symbols[0] if case.callable_symbols else None
        if not feedback:
            return SuiteFreeDecision(
                action="probe",
                code="import buggy\nprint(len(dir(buggy)))",
                predicted_function=fn,
            )
        return SuiteFreeDecision(action="repair", predicted_function=fn)

    def repair(
        self,
        case: SuiteCase,
        source: str,
        failure: Observation,
        diagnosis_summary: str,
        feedback: str,
        seed: int,
    ) -> SuiteRepairProposal:
        self.last_usage = CallUsage()
        fn = case.callable_symbols[0] if case.callable_symbols else None
        if case.case_id in self.repair_rules:
            old, new = self.repair_rules[case.case_id]
            if old in source:
                return SuiteRepairProposal(Patch(source.replace(old, new, 1)), fn)
        raise NotImplementedError(
            f"MockSuiteAdapter has no hardcoded repair rule for {case.case_id!r}; "
            f"use --provider openai for real benchmark runs"
        )


def as_suite_provider(provider: Any) -> SuiteProvider:
    if isinstance(provider, OpenAICompatibleProvider):
        return OpenAISuiteAdapter(provider)
    if isinstance(provider, MockProvider):
        return MockSuiteAdapter()
    return provider


def _append_line(existing: str, line: str) -> str:
    return f"{existing}\n{line}" if existing else line


def run_suite_case(
    case: SuiteCase,
    strategy: Strategy,
    config: RunConfig,
    provider: Any,
    logger: TraceLogger,
    *,
    seed: int = 0,
    sandbox: str = "auto",
    docker_image: str = "active-debug-pysnooper-1",
    wall_clock_limit: float | None = None,
) -> SuiteRunOutcome:
    """Execute one (case, strategy, seed) run and return full telemetry for runs.jsonl."""
    suite_provider = as_suite_provider(provider)
    started = monotonic()
    logger.log(
        "start",
        task=case.case_id,
        revision=case.revision,
        seed=seed,
        strategy=str(strategy),
        config=asdict(config),
        provider=type(provider).__name__,
    )

    actions = Actions(config.max_actions, logger, wall_clock_limit=wall_clock_limit)
    executor = SuiteToolExecutor(
        case.initial_source,
        case.public_suite_dir,
        config.timeout,
        sandbox=sandbox,
        docker_image=docker_image,
    )

    predicted_function: str | None = None
    diagnosis_summary = ""
    feedback = ""

    try:
        actual = actions.call("reproduce", executor.run_public_suite)
        if (not actual.timed_out) and actual.returncode == 0:
            elapsed = monotonic() - started
            logger.log(
                "finish",
                case_id=case.case_id,
                strategy=str(strategy),
                status="completed",
                stop_reason="already_passing",
                actions=actions.used,
                seconds=elapsed,
                predicted_function=None,
            )
            return SuiteRunOutcome(
                case_id=case.case_id,
                revision=case.revision,
                seed=seed,
                strategy=strategy,
                status="completed",
                stop_reason="already_passing",
                actions=actions.used,
                seconds=elapsed,
                final_source=executor.source,
                source_modified=executor.source_modified,
                predicted_function=None,
            )

        if strategy in (Strategy.LLM, Strategy.EIG):
            if actions.remaining >= REPAIR_ACTION_RESERVE + 1:
                plan = actions.call(
                    "hypothesize",
                    suite_provider.hypothesize,
                    case,
                    executor.source,
                    actual,
                    seed,
                )
                state = hypothesis_manager.initialize(
                    [Hypothesis(h.name, h.prior) for h in plan.hypotheses]
                )
                candidate_exps = [
                    executor.register_probe(p.code, p.predictions, p.cost) for p in plan.probes
                ]
                probe_codes = {
                    exp.argument: executor.probe_code(exp) for exp in candidate_exps
                }
                logger.log(
                    "prior",
                    state=asdict(state),
                    hypotheses_detail=[asdict(h) for h in plan.hypotheses],
                    probes_detail=[
                        {"probe_id": exp.argument, "code": probe_codes[exp.argument]}
                        for exp in candidate_exps
                    ],
                )

                used_ids: set[int] = set()
                while not stopping_policy.should_stop(state, config.confidence):
                    options = [e for e in candidate_exps if e.argument not in used_ids]
                    required = 2 if strategy == Strategy.LLM else 1
                    if not options or actions.remaining < REPAIR_ACTION_RESERVE + required:
                        break
                    if strategy == Strategy.LLM:
                        chosen = actions.call(
                            "choose",
                            suite_provider.choose_probe,
                            case,
                            state,
                            plan,
                            options,
                            probe_codes,
                            seed,
                        )
                        if chosen not in options:
                            matches = [o for o in options if o.argument == chosen.argument]
                            if len(matches) != 1:
                                raise ValueError("provider selected an unknown probe option")
                            chosen = matches[0]
                    else:
                        chosen = experiment_selector.select(options, state, config.cost_weight)

                    used_ids.add(chosen.argument)
                    obs = actions.call("probe", executor.run_experiment, chosen)
                    state = belief_updater.update(state, chosen, obs)
                    logger.log(
                        "update",
                        probe=asdict(chosen),
                        probe_code=probe_codes[chosen.argument],
                        observation=asdict(obs),
                        beliefs=[asdict(h) for h in state.hypotheses],
                    )

                best_hyp = max(state.hypotheses, key=lambda h: h.belief)
                spec_map = {h.name: h for h in plan.hypotheses}
                if best_hyp.name in spec_map:
                    predicted_function = spec_map[best_hyp.name].target_function
                diagnosis_summary = _format_suite_diagnosis_summary(state, plan, probe_codes)

        free_probing_done = False
        while True:
            if (
                strategy == Strategy.FREE
                and not free_probing_done
                and actions.remaining >= REPAIR_ACTION_RESERVE + 2
            ):
                decision = actions.call(
                    "free_choice",
                    suite_provider.free_probe,
                    case,
                    executor.source,
                    actual,
                    feedback,
                    seed,
                )
                if predicted_function is None and not executor.source_modified:
                    predicted_function = decision.predicted_function
                if decision.action == "probe" and decision.code:
                    exp = executor.register_probe(decision.code)
                    obs = actions.call("probe", executor.run_experiment, exp)
                    feedback = _append_line(
                        feedback,
                        f"[probe] code={decision.code!r} -> stdout={obs.stdout.strip()!r}, "
                        f"stderr={obs.stderr.strip()!r}, returncode={obs.returncode}, "
                        f"timed_out={obs.timed_out}",
                    )
                    continue
                free_probing_done = True

            proposal = actions.call(
                "repair",
                suite_provider.repair,
                case,
                executor.source,
                actual,
                diagnosis_summary,
                feedback,
                seed,
            )
            if predicted_function is None and not executor.source_modified:
                predicted_function = proposal.predicted_function

            applied = actions.call("apply", executor.apply_patch, proposal.patch)
            if applied.applied:
                verified = actions.call("public_verify", executor.verify_patch)
                if verified.passed:
                    elapsed = monotonic() - started
                    logger.log(
                        "finish",
                        case_id=case.case_id,
                        strategy=str(strategy),
                        status="completed",
                        stop_reason="public_pass",
                        actions=actions.used,
                        seconds=elapsed,
                        predicted_function=predicted_function,
                    )
                    return SuiteRunOutcome(
                        case_id=case.case_id,
                        revision=case.revision,
                        seed=seed,
                        strategy=strategy,
                        status="completed",
                        stop_reason="public_pass",
                        actions=actions.used,
                        seconds=elapsed,
                        final_source=executor.source,
                        source_modified=executor.source_modified,
                        predicted_function=predicted_function,
                    )
                actual = actions.call("reproduce", executor.run_public_suite)
                feedback = _append_line(feedback, f"[verify_failed]\n{_format_obs(actual)}")
            else:
                feedback = _append_line(feedback, f"[apply_failed] {applied.detail}")

    except WallClockExhausted:
        elapsed = monotonic() - started
        logger.log(
            "finish",
            case_id=case.case_id,
            strategy=str(strategy),
            status="timeout",
            stop_reason="timeout",
            actions=actions.used,
            seconds=elapsed,
            predicted_function=predicted_function,
        )
        return SuiteRunOutcome(
            case_id=case.case_id,
            revision=case.revision,
            seed=seed,
            strategy=strategy,
            status="timeout",
            stop_reason="timeout",
            actions=actions.used,
            seconds=elapsed,
            final_source=executor.source,
            source_modified=executor.source_modified,
            predicted_function=predicted_function,
        )
    except BudgetExhausted:
        elapsed = monotonic() - started
        logger.log(
            "finish",
            case_id=case.case_id,
            strategy=str(strategy),
            status="completed",
            stop_reason="budget_exhausted",
            actions=actions.used,
            seconds=elapsed,
            predicted_function=predicted_function,
        )
        return SuiteRunOutcome(
            case_id=case.case_id,
            revision=case.revision,
            seed=seed,
            strategy=strategy,
            status="completed",
            stop_reason="budget_exhausted",
            actions=actions.used,
            seconds=elapsed,
            final_source=executor.source,
            source_modified=executor.source_modified,
            predicted_function=predicted_function,
        )
    except Exception as exc:
        elapsed = monotonic() - started
        detail = f"{type(exc).__name__}: {exc}"
        logger.log(
            "error",
            case_id=case.case_id,
            strategy=str(strategy),
            status="error",
            stop_reason="error",
            actions=actions.used,
            seconds=elapsed,
            detail=detail,
        )
        return SuiteRunOutcome(
            case_id=case.case_id,
            revision=case.revision,
            seed=seed,
            strategy=strategy,
            status="error",
            stop_reason="error",
            actions=actions.used,
            seconds=elapsed,
            final_source=executor.source,
            source_modified=executor.source_modified,
            predicted_function=predicted_function,
            error_detail=detail,
        )
