import unittest

import pandas as pd

from analysis.finetune_benchmark import metrics


class MetricTests(unittest.TestCase):
    def test_invalid_decisions_are_not_dropped(self):
        frame = pd.DataFrame(
            {
                "arm": ["safety_tev"] * 4,
                "cohort": ["xstest_prompt"] * 4,
                "gold": ["unsafe", "unsafe", "safe", "safe"],
                "label": ["unsafe", None, "unsafe", None],
                "status": ["ok", "invalid_output", "ok", "invalid_output"],
            }
        )
        row = metrics(frame).iloc[0]
        self.assertEqual((row.tp, row.fn, row.fp, row.n), (1, 1, 1, 4))
        self.assertEqual(row.f2, 0.5)
        self.assertEqual(row.accuracy, 0.25)
        self.assertEqual(row.no_decision, 2)

    def test_refusal_is_positive_label(self):
        frame = pd.DataFrame(
            {
                "arm": ["cc_repro"] * 2,
                "cohort": ["wildguard_refusal"] * 2,
                "gold": ["refusal", "not_refusal"],
                "label": ["refusal", "not_refusal"],
                "status": ["ok", "ok"],
            }
        )
        row = metrics(frame).iloc[0]
        self.assertEqual((row.tp, row.fn, row.fp), (1, 0, 0))
        self.assertEqual(row.f2, 1.0)


if __name__ == "__main__":
    unittest.main()
