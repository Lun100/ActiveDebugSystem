import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from active_debug.agent.suite_executor import (
    SuiteToolExecutor,
    extract_callable_symbols,
    load_bundle,
)
from active_debug.schemas import Patch

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
