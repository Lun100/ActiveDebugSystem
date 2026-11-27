import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from active_debug import cli
from active_debug.agent.batch_runner import run_bundle_experiment
from active_debug.agent.llm_client import OpenAICompatibleProvider
from active_debug.agent.suite_executor import (
    SuiteToolExecutor,
    extract_callable_symbols,
    load_bundle,
)
from active_debug.agent.suite_runner import (
    HypothesisSpec,
    MockSuiteAdapter,
    ProbeSpec,
    SuiteDiagnosisPlan,
    SuiteRepairProposal,
    normalize_predicted_function,
    run_suite_case,
)
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import (
    Experiment,
    Observation,
    Patch,
    PatchResult,
    RunConfig,
    Strategy,
    TestResult as SchemaTestResult,
)

EXTERNAL_V3_PATH = Path(__file__).resolve().parents[2] / "evaluation-data" / "bugsinpy-focused-v3"


def create_mini_bundle(root: Path) -> Path:
    bundle = root / "mini-bundle"
    src_dir = bundle / "demo-1" / "source"
    pub_dir = bundle / "demo-1" / "public"
    src_dir.mkdir(parents=True)
    pub_dir.mkdir(parents=True)

    (src_dir / "buggy.py").write_text(
        "class Adder:\n"
        "    def compute(self, a, b=1):\n"
        "        return a - b\n\n"
        "def add_one(x):\n"
        "    return Adder().compute(x, 1)\n",
        encoding="utf-8",
    )
    (pub_dir / "test_public.py").write_text(
        "from buggy import add_one\n\n"
        "def test_add_one_tmp(tmp_path):\n"
        "    p = tmp_path / 'val.txt'\n"
        "    p.write_text('5', encoding='utf-8')\n"
        "    assert add_one(int(p.read_text(encoding='utf-8'))) == 6\n",
        encoding="utf-8",
    )
    manifest = {
        "name": "mini-bundle",
        "dataset_revision": "rev-001",
        "evaluation_python": "3.11",
        "image": "sha256:dummy",
        "cases": [
            {
                "case_id": "demo/1",
                "topic": "addition sign bug",
                "revision": "abc123rev",
                "source": "demo-1/source",
                "public_suite": "demo-1/public",
                "difficulty": "basic",
                "functions": 2,
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


class SuiteExecutorTests(unittest.TestCase):
    def test_load_mini_bundle_probe_and_patch(self):
        with TemporaryDirectory() as tmp:
            bundle_path = create_mini_bundle(Path(tmp))
            manifest = load_bundle(bundle_path)
            self.assertEqual(len(manifest.cases), 1)
            case = manifest.cases[0]
            self.assertEqual(case.callable_symbols, ("Adder.compute", "add_one"))

            executor = SuiteToolExecutor(
                case.initial_source, case.public_suite_dir, timeout=2.0, sandbox="local"
            )
            self.assertFalse(executor.source_modified)

            # Initial public suite must fail
            initial_obs = executor.run_public_suite()
            self.assertNotEqual(initial_obs.returncode, 0)

            # Register and run a Python probe snippet
            probe = executor.register_probe(
                "from buggy import Adder\nprint(Adder().compute(5, 2))",
                {"subtracts": "3", "adds": "7"},
            )
            probe_obs = executor.run_experiment(probe)
            self.assertEqual(probe_obs.returncode, 0)
            self.assertEqual(probe_obs.stdout.strip(), "3")

            # Invalid patch preserves original source
            bad = executor.apply_patch(Patch("def broken(:"))
            self.assertFalse(bad.applied)
            self.assertFalse(executor.source_modified)

            # Valid repair makes public suite pass
            fixed = case.initial_source.replace("return a - b", "return a + b")
            good = executor.apply_patch(Patch(fixed))
            self.assertTrue(good.applied)
            self.assertTrue(executor.source_modified)
            self.assertTrue(executor.verify_patch().passed)

    @unittest.skipUnless(EXTERNAL_V3_PATH.is_dir(), "external v3 bundle not present")
    def test_external_v3_bundle_all_12_cases_reproduce_public_failure(self):
        manifest = load_bundle(EXTERNAL_V3_PATH)
        self.assertEqual(len(manifest.cases), 12)
        for case in manifest.cases:
            with self.subTest(case_id=case.case_id):
                self.assertGreaterEqual(len(case.callable_symbols), 1)
                executor = SuiteToolExecutor(
                    case.initial_source, case.public_suite_dir, timeout=3.0, sandbox="local"
                )
                obs = executor.run_public_suite()
                self.assertFalse(obs.timed_out)
                self.assertNotEqual(
                    obs.returncode,
                    0,
                    f"expected buggy public suite to fail for {case.case_id}",
                )


class SuiteRunnerFourStrategiesTests(unittest.TestCase):
    def test_four_strategies_with_mock_and_openai_suite_adapters(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_mini_bundle(root)
            case = load_bundle(bundle).cases[0]

            # 1. MockSuiteAdapter across all 4 strategies
            for strategy in Strategy:
                trace = TraceLogger(root / f"mock-{strategy}.jsonl")
                outcome = run_suite_case(
                    case,
                    strategy,
                    RunConfig(max_actions=12),
                    MockSuiteAdapter(),
                    trace,
                    seed=7,
                    sandbox="local",
                )
                self.assertEqual(outcome.status, "completed")
                self.assertEqual(outcome.stop_reason, "public_pass")
                self.assertTrue(outcome.source_modified)
                self.assertEqual(outcome.predicted_function, "Adder.compute")

            # Verify EIG requires strictly fewer actions than LLM choice on identical plan
            eig_events = TraceLogger(root / "mock-hypothesis_eig.jsonl", allow_append=True).read_events()
            llm_events = TraceLogger(root / "mock-hypothesis_llm.jsonl", allow_append=True).read_events()
            eig_actions = sum(e["event"] == "action" for e in eig_events)
            llm_actions = sum(e["event"] == "action" for e in llm_events)
            self.assertLess(eig_actions, llm_actions)

            # 2. OpenAICompatibleProvider end-to-end on suite EIG with bare method canonicalization
            def fake_transport(req, timeout):
                body = json.loads(req.data.decode("utf-8"))
                assert body.get("seed") == 42
                sys_msg = body["messages"][0]["content"]
                if "fault-diagnosis engine" in sys_msg:
                    content = json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "id": "H_sub",
                                    "function": "compute",  # bare method -> canonicalized to Adder.compute!
                                    "description": "Adder.compute subtracts instead of adding",
                                    "prior": 0.5,
                                },
                                {
                                    "id": "H_arg",
                                    "function": "add_one",
                                    "description": "add_one passes wrong constant",
                                    "prior": 0.5,
                                },
                            ],
                            "probes": [
                                {
                                    "code": "from buggy import Adder\nprint(Adder().compute(5, 2))",
                                    "cost": 1.0,
                                    "predictions": {"H_sub": "3", "H_arg": "7"},
                                }
                            ],
                        }
                    )
                else:
                    content = json.dumps(
                        {
                            "predicted_function": "compute",
                            "source": case.initial_source.replace("return a - b", "return a + b"),
                        }
                    )
                return {
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
                }

            openai_client = OpenAICompatibleProvider(
                api_key="sk-suite-test",
                base_url="https://api.example.com/v1",
                transport=fake_transport,
            )
            eig_trace = TraceLogger(root / "openai-eig.jsonl")
            outcome = run_suite_case(
                case,
                Strategy.EIG,
                RunConfig(max_actions=12),
                openai_client,
                eig_trace,
                seed=42,
                sandbox="local",
            )
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.stop_reason, "public_pass")
            self.assertEqual(outcome.predicted_function, "Adder.compute")

    def test_normalize_predicted_function_canonicalizes_methods(self):
        symbols = ("PeriodicCallback.__init__", "PeriodicCallback._update_next", "helper")
        self.assertEqual(
            normalize_predicted_function("_update_next", symbols),
            "PeriodicCallback._update_next",
        )
        self.assertEqual(normalize_predicted_function("`helper(x)`", symbols), "helper")
        self.assertIsNone(normalize_predicted_function("", symbols))


class FakeSuiteExecutor:
    """In-memory substitute for SuiteToolExecutor, selected by probe code."""

    outputs: dict[str, Observation] = {}

    def __init__(self, source, suite_dir, timeout, **kwargs):
        self.initial_source = source
        self.source = source
        self.timeout = timeout
        self._scripts: dict[int, str] = {}
        self._next = 0

    @property
    def source_modified(self):
        return self.source != self.initial_source

    def register_probe(self, code, predictions=None, cost=1.0):
        probe_id = self._next
        self._next += 1
        self._scripts[probe_id] = code
        return Experiment(probe_id, dict(predictions or {}), cost)

    def probe_code(self, experiment):
        return self._scripts[experiment.argument]

    def run_public_suite(self):
        ok = "return a + b" in self.source
        return Observation("passed" if ok else "failed", "", 0 if ok else 1)

    def run_experiment(self, experiment):
        return self.outputs.get(
            self._scripts[experiment.argument], Observation("", "", 0)
        )

    def apply_patch(self, patch):
        self.source = patch.source
        return PatchResult(True)

    def verify_patch(self):
        ok = "return a + b" in self.source
        return SchemaTestResult(ok, 1, 1)


class _AnomalyAdapter(MockSuiteAdapter):
    def __init__(self):
        super().__init__()
        self.revise_calls = 0

    def hypothesize(self, case, source, failure, seed):
        return SuiteDiagnosisPlan(
            hypotheses=(
                HypothesisSpec("H1", "compute subtracts", 0.5, "Adder.compute"),
                HypothesisSpec("H2", "add_one uses wrong constant", 0.5, "add_one"),
            ),
            probes=(
                ProbeSpec("probe_anomaly", {"H1": "3", "H2": "7"}),
            ),
        )

    def revise(self, case, source, state, plan, probe_codes, anomaly, seed):
        self.revise_calls += 1
        assert anomaly.stdout.strip() == "ERR:TypeError"
        return SuiteDiagnosisPlan(
            hypotheses=(
                HypothesisSpec("H1", "compute subtracts", 0.25, "Adder.compute"),
                HypothesisSpec("H2", "add_one uses wrong constant", 0.25, "add_one"),
                HypothesisSpec("H3_typeerror", "unmodelled TypeError path", 0.5,
                               "Adder.compute"),
            ),
            probes=(
                ProbeSpec("probe_revised", {"H1": "3", "H2": "3", "H3_typeerror": "7"}),
            ),
        )


class _UselessProbeAdapter(MockSuiteAdapter):
    def hypothesize(self, case, source, failure, seed):
        hyps = tuple(HypothesisSpec(f"H{i}", f"cause {i}", 0.25) for i in range(1, 5))
        split = {"H1": "3", "H2": "3", "H3": "7", "H4": "7"}
        return SuiteDiagnosisPlan(
            hypotheses=hyps,
            probes=(
                ProbeSpec("probe_p0", dict(split)),
                ProbeSpec("probe_p1", dict(split)),
                ProbeSpec("probe_p2", {"H1": "3", "H2": "3", "H3": "3", "H4": "3"}),
            ),
        )


class DiagnosisLoopHardeningTests(unittest.TestCase):
    def _run(self, tmp, strategy, adapter):
        bundle = create_mini_bundle(Path(tmp))
        case = load_bundle(bundle).cases[0]
        trace = TraceLogger(Path(tmp) / "trace.jsonl")
        with patch("active_debug.agent.suite_runner.SuiteToolExecutor", FakeSuiteExecutor):
            outcome = run_suite_case(
                case, strategy, RunConfig(max_actions=12), adapter, trace,
                seed=0, sandbox="local",
            )
        return outcome, trace.read_events()

    def test_no_informative_probe_stops_before_burning_actions(self):
        with TemporaryDirectory() as tmp:
            FakeSuiteExecutor.outputs = {
                "probe_p0": Observation("3", "", 0),
                "probe_p1": Observation("3", "", 0),
                "probe_p2": Observation("3", "", 0),
            }
            outcome, events = self._run(tmp, Strategy.EIG, _UselessProbeAdapter())
        self.assertEqual(outcome.stop_reason, "public_pass")
        probes = [
            e for e in events if e["event"] == "action" and e["name"] == "probe"
        ]
        self.assertEqual(len(probes), 1)
        self.assertNotIn("replan", [e["event"] for e in events])

    def test_unpredicted_observation_replans_hypotheses_and_probes(self):
        with TemporaryDirectory() as tmp:
            FakeSuiteExecutor.outputs = {
                "probe_anomaly": Observation("ERR:TypeError", "", 0),
                "probe_revised": Observation("3", "", 0),
            }
            adapter = _AnomalyAdapter()
            outcome, events = self._run(tmp, Strategy.EIG, adapter)
        self.assertEqual(outcome.stop_reason, "public_pass", outcome.error_detail)
        self.assertEqual(adapter.revise_calls, 1)
        names = [e["event"] for e in events]
        self.assertIn("anomaly", names)
        self.assertIn("replan", names)
        prior = next(e for e in events if e["event"] == "prior")
        replan = next(e for e in events if e["event"] == "replan")
        self.assertEqual(len(replan["state"]["hypotheses"]), 3)
        self.assertNotEqual(
            prior["state"]["hypotheses"], replan["state"]["hypotheses"]
        )


class BatchDeliverablesAndCliTests(unittest.TestCase):
    def test_batch_deliverables_plan_runs_candidates_and_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_mini_bundle(root)
            out_dir = root / "deliverables"

            # 1. Run all 4 strategies via CLI --bundle (mock provider repairs demo/1!)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
                rc = cli.main(
                    [
                        "--bundle",
                        str(bundle),
                        "--output",
                        str(out_dir),
                        "--provider",
                        "mock",
                        "--experiment-id",
                        "exp-v3-check",
                        "--model-snapshot",
                        "mock-v1",
                        "--seeds",
                        "0",
                        "--sandbox",
                        "local",
                    ]
                )
            self.assertEqual(rc, 0)
            summary = json.loads(stdout_buf.getvalue())
            self.assertEqual(summary["planned_runs"], 4)
            self.assertEqual(summary["public_pass_runs"], 4)

            # Verify plan.json contract
            plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["experiment_id"], "exp-v3-check")
            self.assertEqual(plan["model_snapshot"], "mock-v1")
            self.assertEqual(plan["max_actions"], 12)
            self.assertEqual(plan["seeds"], [0])

            # Verify runs.jsonl & candidate directories containing buggy.py
            rows = [
                json.loads(line)
                for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["experiment_id"], "exp-v3-check")
                self.assertEqual(row["case_id"], "demo/1")
                self.assertEqual(row["revision"], "abc123rev")
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["stop_reason"], "public_pass")
                self.assertEqual(row["predicted_function"], "Adder.compute")
                self.assertIsNotNone(row["candidate_dir"])
                cand_buggy = out_dir / row["candidate_dir"] / "buggy.py"
                self.assertTrue(cand_buggy.is_file())
                self.assertIn("return a + b", cand_buggy.read_text(encoding="utf-8"))

            # 2. Verify resume_batch skips already-completed runs
            resumed = run_bundle_experiment(
                bundle,
                out_dir,
                RunConfig(),
                MockSuiteAdapter(),
                seeds=(0,),
                experiment_id="exp-v3-check",
                sandbox="local",
                resume_batch=True,
                progress_stream=None,
            )
            self.assertEqual(resumed.executed_runs, 0)
            self.assertEqual(resumed.skipped_completed_runs, 4)

    def test_unmodified_candidate_is_null_and_partial_budget_candidate_is_retained(self):
        class PartialWrongPatchAdapter(MockSuiteAdapter):
            def repair(self, case, source, failure, diagnosis_summary, feedback, seed):
                # Compiles & modifies source, but still fails public test -> hits budget_exhausted
                return SuiteRepairProposal(
                    Patch(source.replace("return a - b", "return a * b")),
                    "Adder.compute",
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_mini_bundle(root)

            # Case A: max_actions=1 -> only reproduce runs, source unmodified -> candidate_dir is null
            out_unmod = root / "out_unmodified"
            run_bundle_experiment(
                bundle,
                out_unmod,
                RunConfig(max_actions=1),
                PartialWrongPatchAdapter(),
                seeds=(0,),
                strategies=(Strategy.DIRECT,),
                sandbox="local",
                progress_stream=None,
            )
            row_unmod = json.loads((out_unmod / "runs.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(row_unmod["status"], "completed")
            self.assertEqual(row_unmod["stop_reason"], "budget_exhausted")
            self.assertIsNone(row_unmod["candidate_dir"])

            # Case B: max_actions=4 -> reproduce(1)+repair(2)+apply(3)+verify(4, fails) -> exhausts
            # Source WAS modified, so candidate_dir MUST be retained with stop_reason="budget_exhausted"
            out_mod = root / "out_modified"
            run_bundle_experiment(
                bundle,
                out_mod,
                RunConfig(max_actions=4),
                PartialWrongPatchAdapter(),
                seeds=(0,),
                strategies=(Strategy.DIRECT,),
                sandbox="local",
                progress_stream=None,
            )
            row_mod = json.loads((out_mod / "runs.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(row_mod["status"], "completed")
            self.assertEqual(row_mod["stop_reason"], "budget_exhausted")
            self.assertIsNotNone(row_mod["candidate_dir"])
            retained_file = out_mod / row_mod["candidate_dir"] / "buggy.py"
            self.assertTrue(retained_file.is_file())
            self.assertIn("return a * b", retained_file.read_text(encoding="utf-8"))
