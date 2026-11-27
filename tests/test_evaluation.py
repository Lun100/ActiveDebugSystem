import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from active_debug.evaluation.dataset_loader import BugsInPyAdapter, load
from active_debug.evaluation.metrics import RunKey, ScoredRun
from active_debug.evaluation.patch_validator import validate, validate_project
from active_debug.evaluation.statistical_analysis import summarize_experiment
from active_debug.evaluation.task_filter import eligible
from active_debug.schemas import Strategy


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.task, held_out = load(Path("examples/simple_bug"))
        self.public_cases = [self.task.public]
        self.all_cases = [self.task.public, *held_out]

    def test_original_code_fails_public_case(self):
        result = validate(self.task.source, self.public_cases, timeout=2)
        self.assertFalse(result.passed)

    def test_public_only_patch_fails_independent_validation(self):
        patch = "def solve(x): return 6"
        self.assertTrue(validate(patch, self.public_cases, timeout=2).passed)
        self.assertFalse(validate(patch, self.all_cases, timeout=2).passed)

    def test_correct_patch_passes_independent_validation(self):
        patch = "def solve(x): return 2 * x"
        result = validate(patch, self.all_cases, timeout=2)
        self.assertTrue(result.passed)

    def test_overlapping_public_and_held_out_inputs_are_rejected(self):
        self.assertFalse(eligible(self.task, self.public_cases))

    def test_project_keeps_held_out_suite_outside_public_suite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            public = root / "public"
            held_out = public / "held-out"
            candidate.mkdir()
            held_out.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "must be separate"):
                validate_project(candidate, public, held_out, image="unused")


class BugsInPyTests(unittest.TestCase):
    def test_public_task_omits_reference_fix(self):
        buggy_revision = "a" * 40
        fixed_revision = "b" * 40
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "projects" / "Example"
            bug = project / "bugs" / "1"
            bug.mkdir(parents=True)
            (project / "project.info").write_text(
                'github_url="https://example.org/example.git"\n', encoding="utf-8"
            )
            (bug / "bug.info").write_text(
                'python_version="3.8.1"\n'
                f'buggy_commit_id="{buggy_revision}"\n'
                f'fixed_commit_id="{fixed_revision}"\n'
                'test_file="tests/test_example.py"\n',
                encoding="utf-8",
            )
            result = asdict(BugsInPyAdapter().load(root, "Example/1"))

        self.assertEqual(result["buggy_revision"], buggy_revision)
        self.assertEqual(result["upstream_test_file"], "tests/test_example.py")
        self.assertNotIn(fixed_revision, result.values())


class PairedStatisticsTests(unittest.TestCase):
    def test_unmatched_runs_stay_in_the_denominator_without_invented_costs(self):
        seed_zero = RunKey("example/1", "old", 0)
        seed_one = RunKey("example/1", "old", 1)
        runs = [
            ScoredRun(seed_zero, Strategy.EIG, "completed", True, "public_pass", actions=5),
            ScoredRun(seed_one, Strategy.DIRECT, "completed", True, "public_pass", actions=3),
        ]
        report = summarize_experiment({seed_zero, seed_one}, runs)
        self.assertFalse(report["complete"])
        self.assertEqual(report["planned_runs"], 8)
        self.assertEqual(report["strategies"][Strategy.EIG]["repair_rate"], 0.5)
        paired = report["eig_comparisons"][Strategy.DIRECT]
        self.assertEqual(paired["observed_pairs"], 0)
        self.assertEqual(paired["repair_rate_difference"], 0.0)
        self.assertIsNone(paired["cost_differences"]["actions"]["mean_difference"])
        self.assertIsNone(paired["repair_rate_difference_ci95"])

    def test_a_duplicate_trial_cannot_be_counted_twice(self):
        key = RunKey("example/1", "old", 0)
        row = ScoredRun(key, Strategy.EIG, "completed", False, "budget_exhausted")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_experiment({key}, [row, row])
