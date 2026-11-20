import unittest
from math import log2

from active_debug.diagnosis.belief_updater import update
from active_debug.diagnosis.experiment_generator import generate as generate_experiments
from active_debug.diagnosis.experiment_selector import information_gain, select
from active_debug.diagnosis.hypothesis_generator import generate as generate_hypotheses
from active_debug.diagnosis.hypothesis_manager import initialize
from active_debug.diagnosis.stopping_policy import should_stop
from active_debug.schemas import (Case, DiagnosisState, Experiment, FailureTrace,
                                  Hypothesis, Observation)


def state_of(*weighted_names: tuple[str, float]) -> DiagnosisState:
    return DiagnosisState([Hypothesis(name, belief) for name, belief in weighted_names])


class InformationGainTests(unittest.TestCase):
    def test_equal_hypotheses_with_same_prediction_gain_zero_bits(self):
        state = state_of(("a", .5), ("b", .5))
        probe = Experiment(0, {"a": "2", "b": "2"})
        self.assertEqual(information_gain(probe, state), 0)

    def test_equal_hypotheses_on_opposite_predictions_gain_one_bit(self):
        state = state_of(("a", .5), ("b", .5))
        probe = Experiment(1, {"a": "1", "b": "2"})
        self.assertEqual(information_gain(probe, state), 1)

    def test_four_equal_hypotheses_binary_split_gain_one_bit(self):
        state = state_of(("a", .25), ("b", .25), ("c", .25), ("d", .25))
        probe = Experiment(0, {"a": "0", "b": "0", "c": "1", "d": "1"})
        self.assertAlmostEqual(information_gain(probe, state), 1, places=12)

    def test_three_equal_hypotheses_uneven_split_matches_hand_calculation(self):
        state = state_of(("a", 1 / 3), ("b", 1 / 3), ("c", 1 / 3))
        probe = Experiment(0, {"a": "0", "b": "1", "c": "1"})
        expected = -(1 / 3 * log2(1 / 3) + 2 / 3 * log2(2 / 3))
        self.assertAlmostEqual(information_gain(probe, state), expected, places=12)

    def test_missing_prediction_names_the_hypothesis(self):
        state = state_of(("a", .5), ("b", .5))
        with self.assertRaisesRegex(ValueError, "b"):
            information_gain(Experiment(0, {"a": "1"}), state)

    def test_select_trades_information_against_cost(self):
        state = state_of(("a", .5), ("b", .5))
        cheap = Experiment(0, {"a": "2", "b": "2"})
        split = Experiment(1, {"a": "1", "b": "2"}, 3)
        self.assertIs(select([cheap, split], state, .1), split)
        self.assertIs(select([cheap, split], state, 1), cheap)

    def test_raising_one_probe_cost_changes_selection(self):
        state = state_of(("a", .5), ("b", .5))
        silent = Experiment(0, {"a": "2", "b": "2"}, 1)
        discriminating = Experiment(1, {"a": "1", "b": "2"}, 1)
        weight = .1
        self.assertIs(select([silent, discriminating], state, weight), discriminating)

        costly = Experiment(1, {"a": "1", "b": "2"}, 20)
        self.assertIs(select([silent, costly], state, weight), silent)

    def test_select_rejects_empty_options(self):
        state = state_of(("a", .5), ("b", .5))
        with self.assertRaisesRegex(ValueError, "candidate"):
            select([], state, .1)

    def test_select_keeps_first_of_equal_scores(self):
        state = state_of(("a", .5), ("b", .5))
        first = Experiment(0, {"a": "0", "b": "1"})
        second = Experiment(1, {"a": "1", "b": "0"})
        self.assertIs(select([first, second], state, 0), first)


class BeliefUpdateTests(unittest.TestCase):
    def test_matching_prediction_concentrates_belief(self):
        state = state_of(("a", .8), ("b", .2))
        probe = Experiment(0, {"a": "1", "b": "2"})
        new = update(state, probe, Observation("2\n", "", 0))
        self.assertEqual([h.belief for h in new.hypotheses], [0, 1])
        self.assertAlmostEqual(sum(h.belief for h in new.hypotheses), 1, places=12)

    def test_shared_prediction_renormalizes_matching_mass(self):
        state = state_of(("a", .5), ("b", .3), ("c", .2))
        probe = Experiment(0, {"a": "x", "b": "x", "c": "y"})
        new = update(state, probe, Observation("x", "", 0))
        beliefs = [h.belief for h in new.hypotheses]
        self.assertAlmostEqual(beliefs[0], .625, places=12)
        self.assertAlmostEqual(beliefs[1], .375, places=12)
        self.assertEqual(beliefs[2], 0)
        self.assertAlmostEqual(sum(beliefs), 1, places=12)

    def test_unexplained_observation_preserves_prior_and_is_recorded(self):
        state = state_of(("a", .8), ("b", .2))
        probe = Experiment(0, {"a": "1", "b": "2"})
        new = update(state, probe, Observation("other", "", 0))
        self.assertEqual(new.hypotheses, state.hypotheses)
        self.assertEqual(new.history, [(probe, Observation("other", "", 0))])
        self.assertEqual(state.history, [])

    def test_error_and_timeout_preserve_prior(self):
        state = state_of(("a", .8), ("b", .2))
        probe = Experiment(0, {"a": "1", "b": "2"})
        for obs in (Observation("1", "", -1), Observation("1", "", -1, True)):
            with self.subTest(obs=obs):
                self.assertEqual(update(state, probe, obs).hypotheses, state.hypotheses)

    def test_inconclusive_observation_skips_prediction_coverage(self):
        state = state_of(("a", .5), ("b", .5))
        probe = Experiment(0, {"a": "1"})
        new = update(state, probe, Observation("1", "", -1, True))
        self.assertEqual(new.hypotheses, state.hypotheses)
        self.assertEqual(len(new.history), 1)

    def test_missing_prediction_on_success_is_an_error(self):
        state = state_of(("a", .5), ("b", .5))
        with self.assertRaisesRegex(ValueError, "b"):
            update(state, Experiment(0, {"a": "1"}), Observation("1", "", 0))

    def test_blank_stdout_can_match_empty_prediction(self):
        state = state_of(("a", .5), ("b", .5))
        probe = Experiment(0, {"a": "", "b": "x"})
        new = update(state, probe, Observation("  \n", "", 0))
        self.assertEqual([h.belief for h in new.hypotheses], [1, 0])

    def test_blank_stdout_without_match_preserves_prior(self):
        state = state_of(("a", .5), ("b", .5))
        probe = Experiment(0, {"a": "1", "b": "2"})
        new = update(state, probe, Observation("", "", 0))
        self.assertEqual(new.hypotheses, state.hypotheses)


class ExperimentGeneratorTests(unittest.TestCase):
    def test_probes_are_ordered_unused_and_cover_every_hypothesis(self):
        state = state_of(("adds_two", .5), ("double_minus_one", .5))
        options = generate_experiments(state)
        self.assertEqual([e.argument for e in options], [3, 1, 2])
        for probe in options:
            self.assertEqual(set(probe.predictions), {"adds_two", "double_minus_one"})
        self.assertEqual(options[0].predictions,
                         {"adds_two": "5", "double_minus_one": "5"})
        self.assertEqual(options[1].predictions,
                         {"adds_two": "3", "double_minus_one": "1"})

    def test_used_arguments_are_not_repeated(self):
        used = Experiment(3, {"adds_two": "5", "double_minus_one": "5"})
        state = DiagnosisState(
            [Hypothesis("adds_two", .5), Hypothesis("double_minus_one", .5)],
            [(used, Observation("5", "", 0))],
        )
        self.assertEqual([e.argument for e in generate_experiments(state)], [1, 2])

    def test_unknown_hypothesis_yields_no_probes(self):
        self.assertEqual(generate_experiments(state_of(("unknown", 1.0))), [])

    def test_mixed_known_and_unknown_hypotheses_yield_no_probes(self):
        state = state_of(("adds_two", .5), ("mystery", .5))
        self.assertEqual(generate_experiments(state), [])


class HypothesisGeneratorTests(unittest.TestCase):
    def test_double_fixture_yields_two_equal_hypotheses(self):
        candidates = generate_hypotheses(
            FailureTrace(Case(3, 6), Observation("5\n", "", 0)))
        self.assertEqual([h.name for h in candidates], ["adds_two", "double_minus_one"])
        self.assertEqual([h.belief for h in candidates], [.5, .5])
        self.assertEqual(sum(h.belief for h in candidates), 1)

    def test_other_traces_fall_back_to_one_unknown_cause(self):
        traces = [
            FailureTrace(Case(3, 6), Observation("7", "", 0)),
            FailureTrace(Case(3, 6), Observation("5", "", 1)),
            FailureTrace(Case(3, 6), Observation("", "timeout", -1, True)),
            FailureTrace(Case(4, 8), Observation("6", "", 0)),
        ]
        for trace in traces:
            with self.subTest(trace=trace):
                self.assertEqual(generate_hypotheses(trace), [Hypothesis("unknown", 1.0)])


class ManagerAndStoppingTests(unittest.TestCase):
    def test_initialize_accepts_unique_normalized_candidates(self):
        state = initialize([Hypothesis("a", .5), Hypothesis("b", .5)])
        self.assertEqual([h.name for h in state.hypotheses], ["a", "b"])
        self.assertEqual(state.history, [])

    def test_initialize_allows_zero_belief_candidate(self):
        state = initialize([Hypothesis("a", 1.0), Hypothesis("b", 0)])
        self.assertEqual([h.belief for h in state.hypotheses], [1, 0])

    def test_initialize_rejects_invalid_candidate_sets(self):
        invalid = [
            [],
            [Hypothesis("a", .5), Hypothesis("a", .5)],
            [Hypothesis("a", .5), Hypothesis("b", .4)],
        ]
        for candidates in invalid:
            with self.subTest(candidates=candidates):
                with self.assertRaises(ValueError):
                    initialize(candidates)

    def test_stopping_threshold_is_inclusive(self):
        self.assertTrue(should_stop(state_of(("a", .95), ("b", .05)), .95))
        self.assertTrue(should_stop(state_of(("a", .5), ("b", .5)), .5))
        self.assertFalse(should_stop(state_of(("a", .95), ("b", .05)), .9500001))
        self.assertFalse(should_stop(state_of(("a", .949), ("b", .051)), .95))


class DiagnosisFlowTests(unittest.TestCase):
    def test_demo_failure_converges_after_one_discriminating_probe(self):
        trace = FailureTrace(Case(3, 6), Observation("5", "", 0))
        state = initialize(generate_hypotheses(trace))
        options = generate_experiments(state)

        # x=3 cannot separate the causes and is penalized by its cost.
        self.assertEqual(information_gain(options[0], state), 0)
        probe = select(options, state, .1)
        self.assertEqual(probe.argument, 1)
        self.assertEqual(information_gain(probe, state), 1)

        state = update(state, probe, Observation("3\n", "", 0))
        self.assertEqual([h.belief for h in state.hypotheses], [1, 0])
        self.assertAlmostEqual(sum(h.belief for h in state.hypotheses), 1, places=12)
        self.assertTrue(should_stop(state, .95))
        self.assertEqual(len(state.history), 1)


if __name__ == "__main__":
    unittest.main()
