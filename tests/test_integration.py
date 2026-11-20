import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from active_debug.agent.baseline import run
from active_debug.agent.llm_client import MockProvider
from active_debug.evaluation.benchmark_runner import benchmark
from active_debug.evaluation.dataset_loader import load
from active_debug.evaluation.metrics import summarize
from active_debug.schemas import RunConfig, Strategy


class PipelineTests(unittest.TestCase):
    def test_four_strategies_and_trace(self):
        task, hidden = load(Path("examples/simple_bug"))
        with TemporaryDirectory() as temp:
            results = []
            for strategy in Strategy:
                results.append(benchmark(task, hidden, strategy, RunConfig(), MockProvider(),
                                         run, Path(temp)))
                path, = Path(temp).glob(f"{strategy}-*.jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[-1]["event"], "finish")
                self.assertEqual(rows[2]["value"]["stdout"].strip(), "5")
                self.assertEqual(sum(r["event"] == "action" for r in rows), results[-1].actions)
            self.assertEqual(summarize(results)["success_rate"], 1)
        self.assertEqual(task, load(Path("examples/simple_bug"))[0])

    def test_budget_exhaustion(self):
        task, hidden = load(Path("examples/simple_bug"))
        with TemporaryDirectory() as temp:
            result = benchmark(task, hidden, Strategy.DIRECT, RunConfig(max_actions=1),
                               MockProvider(), run, Path(temp))
        self.assertEqual(result.actions, 1)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertFalse(result.validation.passed)
