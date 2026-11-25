import io
import json
import os
import unittest
from unittest.mock import patch
from urllib import error as urlerror

from active_debug.agent.llm_client import (
    MockProvider,
    OpenAICompatibleProvider,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    create_provider,
)
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import (
    Case,
    DebugTask,
    DiagnosisState,
    Experiment,
    FailureTrace,
    Hypothesis,
    Observation,
    Patch,
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
