import numpy as np
import pandas as pd
import pytest

from analysis.decision_story import cost_stats, endpoint, paired_tests


def fixture():
    return pd.DataFrame(
        [
            dict(
                arm=arm,
                source_dataset="wildguardtest",
                item_id=str(i),
                input_unit="conversation",
                target="response_harm",
                source_gold=gold,
                status=status,
                label=label,
                provider="hosted",
                token_cost_usd=cost,
                adversarial=False,
            )
            for arm in ["jev_choice", "claude_structured"]
            for i, (gold, status, label, cost) in enumerate(
                [
                    ("unsafe", "ok", "safe", 0.001),
                    ("unsafe", "refusal", None, 0.002),
                    ("unsafe", "ok", "unsafe", 0.003),
                    ("safe", "ok", "unsafe", 0.004),
                    ("safe", "api_error", None, None),
                    ("safe", "ok", "safe", 0.005),
                ]
            )
        ]
    )


def test_no_decision_is_not_silently_assumed_pass_or_block():
    f = fixture()
    # A missing verdict is a detection miss on an unsafe example, but is not a
    # confident safe verdict and does not assert fail-open production behavior.
    assert endpoint(f, "wildguard_response", "miss").event.sum() == 4
    assert endpoint(f, "wildguard_response", "false_safe").event.sum() == 2
    assert endpoint(f, "wildguard_response", "false_alarm").event.sum() == 2
    assert endpoint(f, "wildguard_response", "no_verdict").event.sum() == 4


def test_missing_cost_is_exposed_and_not_imputed_as_zero():
    costs = cost_stats(fixture(), "wildguard_response")
    assert (costs.n == 5).all() and (costs.total_calls == 6).all()
    assert np.allclose(costs.dollars_per_10000, 30)
    assert (costs.usage_coverage == 5 / 6).all()


def test_paired_comparison_requires_the_same_examples():
    f = fixture()
    result = paired_tests(f, [("wildguard_response", "miss", "all")])
    assert result.p_holm.iloc[0] == 1
    with pytest.raises(AssertionError):
        paired_tests(f.drop(index=0), [("wildguard_response", "miss", "all")])
