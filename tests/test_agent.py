import unittest
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import Case, Experiment, Patch


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
