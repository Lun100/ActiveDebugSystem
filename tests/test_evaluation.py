import unittest
from pathlib import Path
from active_debug.evaluation.dataset_loader import load
from active_debug.evaluation.patch_validator import validate
from active_debug.evaluation.task_filter import eligible


class EvaluationTests(unittest.TestCase):
    def test_public_overfit_fails(self):
        task, hidden = load(Path("examples/simple_bug"))
        self.assertFalse(eligible(task, [task.public]))
        self.assertFalse(validate("def solve(x): return 6", [task.public, *hidden], 2).passed)
