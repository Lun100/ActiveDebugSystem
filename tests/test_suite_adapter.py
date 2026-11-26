import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from active_debug.agent.llm_client import OpenAICompatibleProvider
from active_debug.agent.suite_executor import (
    SuiteToolExecutor,
    extract_callable_symbols,
    load_bundle,
)
from active_debug.agent.suite_runner import (
    MockSuiteAdapter,
    normalize_predicted_function,
    run_suite_case,
)
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import Patch, RunConfig, Strategy

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
