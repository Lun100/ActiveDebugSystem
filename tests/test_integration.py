import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from active_debug import cli
from active_debug.agent.baseline import run
from active_debug.agent.llm_client import MockProvider, OpenAICompatibleProvider
from active_debug.evaluation.benchmark_runner import benchmark
from active_debug.evaluation.dataset_loader import load
from active_debug.evaluation.metrics import summarize
from active_debug.schemas import Patch, RunConfig, Strategy


def _make_scripted_openai_provider() -> OpenAICompatibleProvider:
    """Simulated OpenAI-compatible backend handling choose, free_probe, and repair."""
    free_seen = 0

    def transport(req, timeout):
        nonlocal free_seen
        payload = json.loads(req.data.decode("utf-8"))
        system = payload["messages"][0]["content"]
        if "selecting the most informative probe" in system:
            content = '{"argument": 1}'
        elif "Decide whether to run one more integer input probe" in system:
            free_seen += 1
            content = '{"action": "probe", "argument": 1}' if free_seen == 1 else '{"action": "repair"}'
        else:
            content = '```python\ndef solve(x: int) -> int:\n    """Return twice x."""\n    return 2 * x\n```'
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    return OpenAICompatibleProvider(
        api_key="sk-integration-test",
        base_url="https://api.example.com/v1",
        model="gpt-test",
        transport=transport,
    )


class PipelineTests(unittest.TestCase):
    def test_four_strategies_and_trace(self):
        task, hidden = load(Path("examples/simple_bug"))
        with TemporaryDirectory() as temp:
            results = []
            for strategy in Strategy:
                results.append(
                    benchmark(
                        task, hidden, strategy, RunConfig(), MockProvider(), run, Path(temp)
                    )
                )
                (path,) = Path(temp).glob(f"{strategy}-*.jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[-1]["event"], "finish")
                self.assertEqual(rows[2]["value"]["stdout"].strip(), "5")
                self.assertEqual(sum(r["event"] == "action" for r in rows), results[-1].actions)
            self.assertEqual(summarize(results)["success_rate"], 1)
        self.assertEqual(task, load(Path("examples/simple_bug"))[0])

    def test_four_strategies_with_openai_compatible_provider_and_usage_logs(self):
        task, hidden = load(Path("examples/simple_bug"))
        with TemporaryDirectory() as temp:
            results = []
            for strategy in Strategy:
                provider = _make_scripted_openai_provider()
                res = benchmark(task, hidden, strategy, RunConfig(), provider, run, Path(temp))
                results.append(res)
                self.assertTrue(res.validation.passed)
                self.assertEqual(res.stop_reason, "public_pass")

                (path,) = Path(temp).glob(f"{strategy}-*.jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                usage_rows = [r for r in rows if r["event"] == "response" and "usage" in r]
                self.assertGreaterEqual(len(usage_rows), 1)
                self.assertEqual(
                    sum(r["usage"]["total_tokens"] for r in usage_rows),
                    provider.total_tokens,
                )
            self.assertEqual(summarize(results)["success_rate"], 1.0)

    def test_budget_exhaustion(self):
        task, hidden = load(Path("examples/simple_bug"))
        with TemporaryDirectory() as temp:
            result = benchmark(
                task,
                hidden,
                Strategy.DIRECT,
                RunConfig(max_actions=1),
                MockProvider(),
                run,
                Path(temp),
            )
        self.assertEqual(result.actions, 1)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertFalse(result.validation.passed)

    def test_timeout_and_invalid_patch_acceptance_cases(self):
        task, hidden = load(Path("examples/simple_bug"))

        class TimeoutThenFixProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def repair(self, task_arg, failure, state, feedback):
                self.calls += 1
                if self.calls == 1:
                    return Patch("def solve(x: int) -> int:\n    while True:\n        pass\n")
                return Patch("def solve(x: int) -> int:\n    return 2 * x\n")

        with TemporaryDirectory() as temp:
            res = benchmark(
                task,
                hidden,
                Strategy.DIRECT,
                RunConfig(max_actions=10, timeout=0.12),
                TimeoutThenFixProvider(),
                run,
                Path(temp),
            )
            self.assertTrue(res.validation.passed)
            (trace_path,) = Path(temp).glob("direct_repair-*.jsonl")
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertTrue(
                any(
                    r["event"] == "response"
                    and isinstance(r.get("value"), dict)
                    and r["value"].get("timed_out") is True
                    for r in rows
                )
            )

        class AlwaysSyntaxErrorProvider(MockProvider):
            def repair(self, task_arg, failure, state, feedback):
                return Patch("def broken(:")

        with TemporaryDirectory() as temp:
            # 3 actions = reproduce(1) + repair(2) + apply(3, fails); original source preserved
            res_invalid = benchmark(
                task,
                hidden,
                Strategy.DIRECT,
                RunConfig(max_actions=3),
                AlwaysSyntaxErrorProvider(),
                run,
                Path(temp),
            )
            self.assertEqual(res_invalid.actions, 3)
            self.assertEqual(res_invalid.stop_reason, "budget_exhausted")
            self.assertFalse(res_invalid.validation.passed)

    def test_cli_checkpoint_and_resume_end_to_end(self):
        with TemporaryDirectory() as temp:
            ckpt = Path(temp) / "session.ckpt.json"
            out_dir = Path(temp) / "traces"

            buf1 = io.StringIO()
            with patch("sys.stdout", buf1):
                code1 = cli.main(
                    [
                        "--strategy",
                        "direct_repair",
                        "--max-actions",
                        "2",
                        "--checkpoint",
                        str(ckpt),
                        "--output",
                        str(out_dir),
                    ]
                )
            self.assertEqual(code1, 1)
            verdict1 = json.loads(buf1.getvalue())
            self.assertEqual(verdict1["stop_reason"], "budget_exhausted")
            self.assertEqual(verdict1["actions"], 2)

            buf2 = io.StringIO()
            with patch("sys.stdout", buf2):
                code2 = cli.main(
                    [
                        "--strategy",
                        "direct_repair",
                        "--max-actions",
                        "6",
                        "--checkpoint",
                        str(ckpt),
                        "--resume",
                        "--output",
                        str(out_dir),
                    ]
                )
            self.assertEqual(code2, 0)
            verdict2 = json.loads(buf2.getvalue())
            self.assertEqual(verdict2["stop_reason"], "public_pass")
            self.assertTrue(verdict2["validation"]["passed"])
            self.assertEqual(verdict2["actions"], 4)
