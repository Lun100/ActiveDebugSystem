import io
import json
import os
import unittest
from unittest.mock import patch
from urllib import error as urlerror

from pathlib import Path
from tempfile import TemporaryDirectory

from active_debug.agent.agent import Actions, BudgetExhausted, WallClockExhausted
from active_debug.agent.baseline import run
from active_debug.agent.debug_controller import diagnose
from active_debug.agent.llm_client import (
    MockProvider,
    OpenAICompatibleProvider,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    create_provider,
)
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import (
    Case,
    DebugTask,
    DiagnosisState,
    Experiment,
    FailureTrace,
    Hypothesis,
    Observation,
    Patch,
    RunConfig,
    Strategy,
)


def _chat_payload(content: str, prompt_tokens: int = 12, completion_tokens: int = 8) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class ToolTests(unittest.TestCase):
    def test_exception_and_timeout(self):
        executor = ToolExecutor("def solve(x): return 1/0", Case(1, 2), 2)
        obs = executor.run_experiment(Experiment(1))
        self.assertNotEqual(obs.returncode, 0)
        self.assertIn("ZeroDivisionError", obs.stderr)
        executor.apply_patch(Patch("def solve(x):\n    while True: pass"))
        executor.timeout = 0.15
        self.assertTrue(executor.run_experiment(Experiment(1)).timed_out)

    def test_invalid_patch_preserves_source(self):
        executor = ToolExecutor("def solve(x): return x", Case(1, 1), 2)
        self.assertFalse(executor.apply_patch(Patch("def ?")).applied)
        self.assertTrue(executor.verify_patch().passed)
        # Missing solve() entry point is also rejected without overwriting source
        missing_entry = executor.apply_patch(Patch("def helper(x):\n    return x * 2\n"))
        self.assertFalse(missing_entry.applied)
        self.assertIn("solve", missing_entry.detail)
        self.assertTrue(executor.verify_patch().passed)

    def test_output_truncation_cap(self):
        noisy_src = "def solve(x):\n    print('A' * 500)\n    return x\n"
        executor = ToolExecutor(noisy_src, Case(1, 1), 2, max_output_chars=128)
        obs = executor.run_experiment(Experiment(1))
        self.assertEqual(obs.returncode, 0)
        self.assertIn("...[truncated]", obs.stdout)
        self.assertLessEqual(len(obs.stdout), 150)


class ActionsAndLoggerTests(unittest.TestCase):
    def test_budget_telemetry_error_logging_and_redaction(self):
        secret = "sk-log-secret-999"
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": secret}, clear=False
        ):
            log_path = Path(tmp) / "trace.jsonl"
            logger = TraceLogger(log_path)
            # Exclusive creation by default
            with self.assertRaises(FileExistsError):
                TraceLogger(log_path)
            # Append mode works for resumed traces
            resumed_logger = TraceLogger(log_path, allow_append=True)

            provider = OpenAICompatibleProvider(
                api_key=secret,
                base_url="https://api.example.com/v1",
                transport=lambda req, timeout: _chat_payload('{"action": "repair"}', 7, 3),
            )
            actions = Actions(2, resumed_logger)
            task = DebugTask("t", "def solve(x): return x", Case(1, 1))
            failure = FailureTrace(Case(1, 1), Observation("0", "", 0))

            res = actions.call("free_choice", provider.free_probe, task, failure, secret)
            self.assertIsNone(res)
            self.assertEqual(actions.remaining, 1)
            self.assertEqual(actions.total_tokens, 10)

            # Failing call still deducts action budget before invocation and logs action_error
            def boom():
                raise RuntimeError(f"boom {secret}")

            with self.assertRaises(RuntimeError):
                actions.call("explode", boom)
            self.assertEqual(actions.used, 2)

            with self.assertRaises(BudgetExhausted):
                actions.call("over_budget", lambda: 1)

            raw_text = log_path.read_text(encoding="utf-8")
            self.assertNotIn(secret, raw_text)
            events = logger.read_events()
            self.assertEqual([e["event"] for e in events], ["action", "response", "action", "action_error"])
            self.assertEqual(events[1]["usage"]["total_tokens"], 10)
            self.assertIn("duration_ms", events[1])

    def test_wall_clock_deadline(self):
        with TemporaryDirectory() as tmp:
            logger = TraceLogger(Path(tmp) / "clock.jsonl")
            actions = Actions(5, logger, wall_clock_limit=0.01)
            import time

            time.sleep(0.02)
            with self.assertRaises(WallClockExhausted):
                actions.call("late", lambda: 42)
            self.assertEqual(actions.used, 0)


class ControllerAndBaselineTests(unittest.TestCase):
    def setUp(self):
        self.task = DebugTask(
            "double",
            'def solve(x: int) -> int:\n    """Return twice x."""\n    return x + 2\n',
            Case(3, 6),
        )

    def test_already_passing_stops_after_reproduce(self):
        passing_task = DebugTask("ok", "def solve(x: int) -> int:\n    return 2 * x\n", Case(3, 6))
        with TemporaryDirectory() as tmp:
            logger = TraceLogger(Path(tmp) / "ok.jsonl")
            actions = Actions(12, logger)
            tools = ToolExecutor(passing_task.source, passing_task.public, 2.0)
            reason = run(passing_task, Strategy.EIG, RunConfig(), actions, tools, MockProvider())
            self.assertEqual(reason, "already_passing")
            self.assertEqual(actions.used, 1)

    def test_diagnose_rejects_unknown_provider_choice(self):
        class BadChoiceProvider(MockProvider):
            def choose(self, state, options):
                return Experiment(999)

        with TemporaryDirectory() as tmp:
            logger = TraceLogger(Path(tmp) / "bad.jsonl")
            actions = Actions(12, logger)
            tools = ToolExecutor(self.task.source, self.task.public, 2.0)
            failure = FailureTrace(Case(3, 6), Observation("5\n", "", 0))
            with self.assertRaises(ValueError):
                diagnose(failure, Strategy.LLM, RunConfig(), actions, tools, BadChoiceProvider())

    def test_iterative_repair_recovers_from_invalid_and_wrong_patches(self):
        seen_feedback: list[str] = []

        class MultiTurnRepairProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.turn = 0

            def repair(self, task, failure, state, feedback):
                seen_feedback.append(feedback)
                self.turn += 1
                if self.turn == 1:
                    return Patch("def broken(:")  # syntax error -> apply fails
                if self.turn == 2:
                    return Patch("def solve(x):\n    return x + 1\n")  # wrong output -> verify fails
                return Patch("def solve(x):\n    return 2 * x\n")

        with TemporaryDirectory() as tmp:
            logger = TraceLogger(Path(tmp) / "iter.jsonl")
            actions = Actions(12, logger)
            tools = ToolExecutor(self.task.source, self.task.public, 2.0)
            reason = run(
                self.task, Strategy.DIRECT, RunConfig(), actions, tools, MultiTurnRepairProvider()
            )
            self.assertEqual(reason, "public_pass")
            self.assertEqual(seen_feedback[0], "")
            self.assertIn("[apply_failed]", seen_feedback[1])
            self.assertIn("[verify_failed]", seen_feedback[2])

    def test_free_debug_reserves_budget_for_repair_when_tight(self):
        class GreedyProbeProvider(MockProvider):
            def free_probe(self, task, failure, feedback):
                return Experiment(1)  # would probe forever if allowed

        with TemporaryDirectory() as tmp:
            logger = TraceLogger(Path(tmp) / "tight.jsonl")
            # 4 actions = 1 reproduce + 3 (repair, apply, public_verify); no room for free_choice+probe
            actions = Actions(4, logger)
            tools = ToolExecutor(self.task.source, self.task.public, 2.0)
            reason = run(
                self.task,
                Strategy.FREE,
                RunConfig(max_actions=4),
                actions,
                tools,
                GreedyProbeProvider(),
            )
            self.assertEqual(reason, "public_pass")
            self.assertEqual(actions.used, 4)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.task = DebugTask(
            "double",
            'def solve(x: int) -> int:\n    """Return twice x."""\n    return x + 2\n',
            Case(3, 6),
        )
        self.failure = FailureTrace(Case(3, 6), Observation("5\n", "", 0))
        self.state = DiagnosisState(
            [Hypothesis("adds_two", 0.9), Hypothesis("double_minus_one", 0.1)],
            [(Experiment(1, {"adds_two": "3", "double_minus_one": "1"}), Observation("3\n", "", 0))],
        )

    def test_config_validation_and_secret_redaction(self):
        with self.assertRaises(ProviderConfigError):
            OpenAICompatibleProvider(api_key="", base_url="https://api.example.com/v1")

        # Local endpoints (e.g. Ollama / vLLM) do not require an API key
        local = OpenAICompatibleProvider(api_key=None, base_url="http://localhost:11434/v1")
        self.assertIsNone(local.last_usage)

        secret = "sk-super-secret-token-12345"

        def failing_transport(req, timeout):
            raise urlerror.URLError(f"failed connecting with {secret}")

        provider = OpenAICompatibleProvider(
            api_key=secret,
            base_url="https://api.example.com/v1",
            max_retries=0,
            transport=failing_transport,
        )
        self.assertNotIn(secret, repr(provider))
        with self.assertRaises(ProviderError) as ctx:
            provider.free_probe(self.task, self.failure, "")
        self.assertNotIn(secret, str(ctx.exception))

    def test_choose_free_probe_and_repair_parsing(self):
        captured_prompts: list[str] = []
        responses = [
            _chat_payload('```json\n{"argument": 1}\n```', 10, 5),
            _chat_payload('{"action": "probe", "argument": 4}', 15, 6),
            _chat_payload('{"action": "repair"}', 16, 4),
            _chat_payload('```python\ndef solve(x: int) -> int:\n    return 2 * x\n```', 20, 10),
        ]

        def fake_transport(req, timeout):
            self.assertEqual(req.get_header("Authorization"), "Bearer sk-test")
            body = json.loads(req.data.decode("utf-8"))
            captured_prompts.append(body["messages"][1]["content"])
            return responses.pop(0)

        provider = OpenAICompatibleProvider(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            transport=fake_transport,
        )
        options = [
            Experiment(3, {"adds_two": "5", "double_minus_one": "5"}),
            Experiment(1, {"adds_two": "3", "double_minus_one": "1"}),
        ]
        chosen = provider.choose(self.state, options)
        self.assertEqual(chosen, options[1])

        probe = provider.free_probe(self.task, self.failure, "")
        self.assertEqual(probe, Experiment(4))
        self.assertIsNone(provider.free_probe(self.task, self.failure, "saw 4->6"))

        patch_obj = provider.repair(self.task, self.failure, self.state, "retry feedback")
        self.assertIn("return 2 * x", patch_obj.source)
        # Diagnosis posterior and history must be injected into repair prompt
        self.assertIn("adds_two: 0.9000", captured_prompts[-1])
        self.assertIn("retry feedback", captured_prompts[-1])

        summary = provider.usage_summary()
        self.assertEqual(summary["total_calls"], 4)
        self.assertEqual(summary["total_prompt_tokens"], 61)
        self.assertEqual(summary["total_completion_tokens"], 25)
        self.assertEqual(summary["total_tokens"], 86)

    def test_retry_on_transient_http_error_and_reject_unknown_choice(self):
        calls = 0

        def flaky_transport(req, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urlerror.HTTPError(
                    req.full_url, 429, "Too Many Requests", hdrs=None, fp=io.BytesIO(b"rate limit")
                )
            return _chat_payload('{"argument": 99}')

        provider = OpenAICompatibleProvider(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            max_retries=2,
            retry_backoff=0.0,
            transport=flaky_transport,
            sleep_fn=lambda _: None,
        )
        with self.assertRaises(ProviderResponseError):
            provider.choose(self.state, [Experiment(1, {"adds_two": "3", "double_minus_one": "1"})])
        self.assertEqual(calls, 2)
        self.assertEqual(provider.total_retries, 1)

    def test_create_provider_from_env(self):
        self.assertIsInstance(create_provider("mock"), MockProvider)
        with patch.dict(
            os.environ,
            {
                "ACTIVE_DEBUG_API_KEY": "sk-env-key",
                "ACTIVE_DEBUG_BASE_URL": "https://llm.example.org/v1",
                "ACTIVE_DEBUG_MODEL": "deepseek-coder",
            },
            clear=False,
        ):
            client = create_provider("openai")
            self.assertIsInstance(client, OpenAICompatibleProvider)
            self.assertEqual(client.model, "deepseek-coder")
            self.assertEqual(client.base_url, "https://llm.example.org/v1")
