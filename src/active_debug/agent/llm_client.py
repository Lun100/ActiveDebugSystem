"""Injectable LLM providers, prompt builders, and zero-dependency OpenAI-compatible client."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any, Protocol
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from active_debug.schemas import (
    DebugTask,
    DiagnosisState,
    Experiment,
    FailureTrace,
    Patch,
)

_RETRYABLE_HTTP_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ProviderError(RuntimeError):
    """Base error for provider configuration, transport, or response parsing failures."""


class ProviderConfigError(ProviderError):
    """Raised when provider credentials or endpoint settings are invalid."""


class ProviderResponseError(ProviderError):
    """Raised when the upstream LLM response fails validation or cannot be parsed."""


@dataclass(frozen=True)
class CallUsage:
    """Token, retry, and wall-clock telemetry for a single provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    retries: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "retries": self.retries,
            "latency_ms": round(self.latency_ms, 3),
        }


class Provider(Protocol):
    def choose(self, state: DiagnosisState, options: list[Experiment]) -> Experiment: ...
    def repair(
        self,
        task: DebugTask,
        failure: FailureTrace,
        state: DiagnosisState | None,
        feedback: str,
    ) -> Patch: ...
    def free_probe(
        self,
        task: DebugTask,
        failure: FailureTrace,
        feedback: str,
    ) -> Experiment | None: ...


class MockProvider:
    """Deterministic offline provider supporting the double demo fixture."""

    def __init__(self) -> None:
        self.last_usage: CallUsage | None = None

    def choose(self, state: DiagnosisState, options: list[Experiment]) -> Experiment:
        if not options:
            raise ValueError("cannot choose from empty experiment options")
        self.last_usage = CallUsage()
        return options[-1]

    def free_probe(
        self,
        task: DebugTask,
        failure: FailureTrace,
        feedback: str,
    ) -> Experiment | None:
        self.last_usage = CallUsage()
        return None if feedback else Experiment(1)

    def repair(
        self,
        task: DebugTask,
        failure: FailureTrace,
        state: DiagnosisState | None,
        feedback: str,
    ) -> Patch:
        self.last_usage = CallUsage()
        if '"""Return twice x."""' not in task.source:
            raise NotImplementedError("Mock repair only supports the double fixture")
        return Patch(task.source.replace("return x + 2", "return 2 * x"))


Transport = Callable[[urlrequest.Request, float], dict[str, Any]]


def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _is_local_endpoint(base_url: str) -> bool:
    parsed = urlparse.urlparse(base_url)
    return (parsed.hostname or "").lower() in _LOCAL_HOSTS


def _redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "***")


def load_dotenv(path: Path = Path(".env"), *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a local .env file without external dependencies."""
    if not path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        val = raw_val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            quote = val[0]
            val = val[1:-1]
            if quote == '"':
                val = val.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")
        else:
            # Strip trailing inline comment for unquoted values
            if " #" in val:
                val = val.split(" #", 1)[0].rstrip()
        loaded[key] = val
        if override or (val and key not in os.environ):
            os.environ[key] = val
    return loaded


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = _CODE_FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"invalid JSON object in model output: {exc}") from exc
    raise ProviderResponseError("model output did not contain a JSON object")


def _extract_python_source(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ProviderResponseError("model returned empty repair output")
    try:
        obj = _extract_json_object(text)
        if isinstance(obj.get("source"), str) and obj["source"].strip():
            return obj["source"]
    except ProviderResponseError:
        pass
    fenced = _CODE_FENCE_RE.findall(text)
    if fenced:
        for block in fenced:
            if "def solve" in block:
                return block
        return fenced[0]
    return text


def _format_failure(failure: FailureTrace) -> str:
    obs = failure.actual
    return (
        f"Public case: solve({failure.case.argument}) expected {failure.case.expected}\n"
        f"Actual stdout: {obs.stdout.strip()!r}, stderr: {obs.stderr.strip()!r}, "
        f"returncode: {obs.returncode}, timed_out: {obs.timed_out}"
    )


def _format_diagnosis_state(state: DiagnosisState) -> str:
    ranked = sorted(state.hypotheses, key=lambda h: h.belief, reverse=True)
    lines = ["Hypotheses (sorted by posterior belief):"]
    for h in ranked:
        lines.append(f"  - {h.name}: {h.belief:.4f}")
    if state.history:
        lines.append("Completed diagnostic probes:")
        for idx, (probe, obs) in enumerate(state.history, start=1):
            lines.append(
                f"  {idx}. solve({probe.argument}) -> stdout={obs.stdout.strip()!r}, "
                f"returncode={obs.returncode}, timed_out={obs.timed_out} "
                f"(predictions={probe.predictions})"
            )
    return "\n".join(lines)


class OpenAICompatibleProvider:
    """Zero-dependency HTTP client for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 90.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        transport: Transport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed = urlparse.urlparse(normalized_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ProviderConfigError(f"invalid base_url: {base_url!r}")
        cleaned_key = (api_key or "").strip()
        if not cleaned_key and not _is_local_endpoint(normalized_url):
            raise ProviderConfigError(
                "missing API key; set ACTIVE_DEBUG_API_KEY or OPENAI_API_KEY in environment"
            )
        if not model or not model.strip():
            raise ProviderConfigError("model name must be non-empty")
        if timeout <= 0:
            raise ProviderConfigError("timeout must be positive")
        if max_retries < 0:
            raise ProviderConfigError("max_retries must be non-negative")

        self._api_key = cleaned_key
        self.base_url = normalized_url
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._transport = transport or self._default_transport
        self._sleep = sleep_fn
        self._ssl_context = _build_ssl_context()

        self.last_usage: CallUsage | None = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_calls = 0
        self.total_retries = 0

    def __repr__(self) -> str:
        masked = "***" if self._api_key else None
        return (
            f"OpenAICompatibleProvider(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key={masked!r})"
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        dotenv_path: Path | None = Path(".env"),
    ) -> OpenAICompatibleProvider:
        if dotenv_path is not None:
            load_dotenv(dotenv_path)
        api_key = os.environ.get("ACTIVE_DEBUG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        resolved_url = (
            base_url
            or os.environ.get("ACTIVE_DEBUG_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        resolved_model = (
            model
            or os.environ.get("ACTIVE_DEBUG_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        return cls(
            api_key=api_key,
            base_url=resolved_url,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def clone(self) -> OpenAICompatibleProvider:
        """Create a thread-isolated provider instance with identical configuration."""
        return OpenAICompatibleProvider(
            api_key=self._api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            transport=None if self._transport == self._default_transport else self._transport,
            sleep_fn=self._sleep,
        )

    def usage_summary(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _default_transport(self, req: urlrequest.Request, timeout: float) -> dict[str, Any]:
        context = self._ssl_context if req.full_url.startswith("https://") else None
        with urlrequest.urlopen(req, timeout=timeout, context=context) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"non-JSON HTTP response from provider: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("provider HTTP response root must be a JSON object")
        return data

    def chat(self, system_prompt: str, user_prompt: str, *, seed: int | None = None) -> str:
        """Execute a single chat completion call with retry and token accounting."""
        return self._chat(system_prompt, user_prompt, seed=seed)

    def _chat(self, system_prompt: str, user_prompt: str, *, seed: int | None = None) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempt = 0
        started = time.monotonic()
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            body = json.dumps(payload).encode("utf-8")
            req = urlrequest.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                response = self._transport(req, self.timeout)
                break
            except urlerror.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    detail = str(exc)
                finally:
                    exc.close()
                # Gracefully drop optional 'seed' parameter if an endpoint rejects it with 400
                if exc.code == 400 and "seed" in payload and "seed" in detail.lower():
                    payload.pop("seed", None)
                    continue
                safe_detail = _redact_secret(detail, self._api_key)
                last_error = ProviderError(f"HTTP {exc.code} from provider: {safe_detail}")
                if exc.code not in _RETRYABLE_HTTP_CODES or attempt >= self.max_retries:
                    raise last_error from None
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                safe_msg = _redact_secret(str(exc), self._api_key)
                last_error = ProviderError(f"provider transport error: {safe_msg}")
                if attempt >= self.max_retries:
                    raise last_error from None
            attempt += 1
            self.total_retries += 1
            if self.retry_backoff > 0:
                self._sleep(self.retry_backoff * (2 ** (attempt - 1)))
        else:  # pragma: no cover - defensive
            raise last_error or ProviderError("provider request failed")

        elapsed_ms = (time.monotonic() - started) * 1000.0
        usage_raw = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        total_tokens = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        comp_details = (
            usage_raw.get("completion_tokens_details")
            if isinstance(usage_raw.get("completion_tokens_details"), dict)
            else {}
        )
        reasoning_tokens = int(comp_details.get("reasoning_tokens") or 0)

        self.last_usage = CallUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            retries=attempt,
            latency_ms=elapsed_ms,
        )
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response missing non-empty 'choices'")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("provider response missing assistant text content")
        return content

    def choose(self, state: DiagnosisState, options: list[Experiment]) -> Experiment:
        if not options:
            raise ValueError("cannot choose from empty experiment options")
        system_prompt = (
            "You are an active debugging agent selecting the most informative probe input. "
            "Respond ONLY with a JSON object of the form {\"argument\": <int>} matching one "
            "of the candidate options."
        )
        option_lines = [
            f"- index={idx}, argument={opt.argument}, cost={opt.cost}, predictions={opt.predictions}"
            for idx, opt in enumerate(options)
        ]
        user_prompt = (
            f"{_format_diagnosis_state(state)}\n\n"
            f"Candidate experiments:\n" + "\n".join(option_lines)
        )
        content = self._chat(system_prompt, user_prompt)
        obj = _extract_json_object(content)

        if "argument" in obj and type(obj["argument"]) is int:
            target = obj["argument"]
            for opt in options:
                if opt.argument == target:
                    return opt
            raise ProviderResponseError(f"selected argument {target} is not among options")

        if "index" in obj and type(obj["index"]) is int:
            idx = obj["index"]
            if 0 <= idx < len(options):
                return options[idx]
            raise ProviderResponseError(f"selected index {idx} out of range")

        raise ProviderResponseError("choose response must contain integer 'argument' or 'index'")

    def free_probe(
        self,
        task: DebugTask,
        failure: FailureTrace,
        feedback: str,
    ) -> Experiment | None:
        system_prompt = (
            "You are debugging a Python function solve(x: int) -> int. "
            "Decide whether to run one more integer input probe or proceed to repair. "
            "Respond ONLY with JSON: {\"action\": \"probe\", \"argument\": <int>} "
            "or {\"action\": \"repair\"}."
        )
        user_prompt = (
            f"Task: {task.name}\n"
            f"Current source:\n```python\n{task.source}\n```\n"
            f"{_format_failure(failure)}\n"
            f"Prior observations / feedback:\n{feedback or '(none)'}"
        )
        content = self._chat(system_prompt, user_prompt)
        obj = _extract_json_object(content)
        action = str(obj.get("action", "")).strip().lower()
        if action == "repair":
            return None
        if action == "probe":
            arg = obj.get("argument")
            if type(arg) is not int:
                raise ProviderResponseError("probe action requires integer 'argument'")
            return Experiment(arg)
        raise ProviderResponseError(f"unsupported free_probe action: {action!r}")

    def repair(
        self,
        task: DebugTask,
        failure: FailureTrace,
        state: DiagnosisState | None,
        feedback: str,
    ) -> Patch:
        system_prompt = (
            "You are an expert Python program repair agent. "
            "Return a complete replacement module defining solve(x: int) -> int. "
            "Respond with either JSON {\"source\": \"...\"} or a single ```python code block."
        )
        sections = [
            f"Task: {task.name}",
            f"Current source:\n```python\n{task.source}\n```",
            _format_failure(failure),
        ]
        if state is not None:
            sections.append(_format_diagnosis_state(state))
        if feedback:
            sections.append(f"Accumulated feedback:\n{feedback}")
        content = self._chat(system_prompt, "\n\n".join(sections))
        source = _extract_python_source(content)
        try:
            return Patch(source)
        except ValueError as exc:
            raise ProviderResponseError(f"invalid patch produced by provider: {exc}") from exc


def create_provider(
    kind: str = "mock",
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> Provider:
    """Instantiate a provider by name without embedding credentials in config."""
    normalized = kind.strip().lower()
    if normalized == "mock":
        return MockProvider()
    if normalized in ("openai", "openai-compatible", "http"):
        return OpenAICompatibleProvider.from_env(
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ProviderConfigError(f"unknown provider kind: {kind!r}")
