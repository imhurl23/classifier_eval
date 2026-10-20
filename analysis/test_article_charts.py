import numpy as np
import pandas as pd

from analysis.article_charts import significant_marks
from analysis.decision_common import pareto_mask


def test_frontier_keeps_equal_points_but_excludes_dominated_models():
    cost = [1, 1, 2, 2, 3]
    quality = [0.8, 0.8, 0.7, 0.9, 0.9]
    np.testing.assert_array_equal(
        pareto_mask(cost, quality), [True, True, False, True, False]
    )


def test_significance_marks_use_the_correct_endpoint_and_adjusted_test():
    rates = pd.DataFrame(
        {"arm": ["b", "a", "c", "d"], "rate": [0.1, 0.1, 0.2, 0.3]}
    ).set_index("arm")
    comparisons = pd.DataFrame(
        [
            {"arm_a": "a", "arm_b": "b", "metric": "miss", "p_holm": 1.0},
            {"arm_a": "c", "arm_b": "a", "metric": "miss", "p_holm": 0.01},
            {"arm_a": "a", "arm_b": "d", "metric": "miss", "p_holm": 0.2},
            {"arm_a": "a", "arm_b": "d", "metric": "false_alarm", "p_holm": 0.01},
        ]
    )
    marks, reference = significant_marks(rates, comparisons, metric="miss")
    assert reference == "a"
    assert marks == ["", "", " †", ""]
