import unittest
from active_debug.diagnosis.belief_updater import update
from active_debug.diagnosis.experiment_selector import information_gain, select
from active_debug.schemas import DiagnosisState, Experiment, Hypothesis, Observation


class DiagnosisTests(unittest.TestCase):
    def test_gain_cost_and_update(self):
        state = DiagnosisState([Hypothesis("a", .5), Hypothesis("b", .5)])
        cheap = Experiment(0, {"a": "2", "b": "2"})
        split = Experiment(1, {"a": "1", "b": "2"}, 3)
        self.assertEqual(information_gain(cheap, state), 0)
        self.assertEqual(information_gain(split, state), 1)
        self.assertEqual(select([cheap, split], state, .1), split)
        self.assertEqual(select([cheap, split], state, 1), cheap)
        for obs in (Observation("other", "", 0), Observation("1", "", -1, True)):
            self.assertEqual(update(state, split, obs).hypotheses, state.hypotheses)
        self.assertEqual(update(state, split, Observation("1", "", 0)).hypotheses[0].belief, 1)
