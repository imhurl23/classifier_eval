import pandas as pd
import pytest

from analysis.failure_report import metrics, outcome, paired_gains, roc
from analysis.safety_pairs import pair_summary, safety_pairs


def test_failure_types_distinguish_classifier_refusal_from_observed_refusal():
    assert outcome("refusal", None, "refusal") == "classifier_refusal"
    assert outcome("ok", "not_refusal", "refusal") == "missed_refusal"
    assert outcome("ok", "refusal", "not_refusal") == "false_refusal"
    assert outcome("ok", "needs_caution", "unsafe") == "caution_abstention"
    assert outcome("truncated", None, "unsafe") == "truncated"
    assert outcome("ok", "safe", None) == "unscored"


def test_strict_miss_denominator_and_disjoint_error_accounting():
    rows = [
        {
            "error": True,
            "outcome": "false_safe",
            "source_gold": "unsafe",
            "status": "ok",
            "predicted_label_confidence": 0.99,
            "high_confidence_error": True,
        },
        {
            "error": True,
            "outcome": "classifier_refusal",
            "source_gold": "unsafe",
            "status": "refusal",
            "predicted_label_confidence": None,
            "high_confidence_error": False,
        },
        {
            "error": False,
            "outcome": "correct",
            "source_gold": "unsafe",
            "status": "ok",
            "predicted_label_confidence": 0.8,
            "high_confidence_error": False,
        },
        {
            "error": True,
            "outcome": "false_unsafe",
            "source_gold": "safe",
            "status": "ok",
            "predicted_label_confidence": 0.7,
            "high_confidence_error": False,
        },
    ]
    m = metrics(pd.DataFrame(rows))
    assert m["errors"] == 3 and m["false_safe"] == 1 and m["classifier_refusal"] == 1
    assert m["unsafe_miss_n"] == 3 and m["unsafe_miss_k"] == 2
    assert m["safe_false_unsafe_n"] == 1 and m["safe_false_unsafe_k"] == 1


def test_paired_finetune_gains_use_same_items_not_marginal_row_counts():
    base = {
        "source_dataset": "test",
        "input_unit": "conversation",
        "target": "interaction_harm",
        "human_gold_available": True,
    }
    rows = [
        {**base, "item_id": str(i), "arm": a, "error": bad}
        for a, i, bad in [
            ("tev1_native", 0, False),
            ("qwen35_4b_letters", 0, True),
            ("tev1_native", 1, True),
            ("qwen35_4b_letters", 1, True),
            ("tev1_native", 2, False),
        ]
    ]
    m = paired_gains(pd.DataFrame(rows)).iloc[0]
    assert (
        m.n == 2
        and m.a_wins == 1
        and m.a_losses == 0
        and m.both_wrong == 1
        and m.both_correct == 0
    )
    assert m.accuracy_delta_pp == 50


def test_probability_roc_groups_ties_and_has_correct_direction():
    assert roc([0.9, 0.8, 0.2, 0.1], [True, True, False, False])[2] == 1
    assert roc([0.1, 0.2, 0.8, 0.9], [True, True, False, False])[2] == 0
    assert roc([0.5, 0.5, 0.5, 0.5], [True, True, False, False])[2] == 0.5


def pair_rows(
    item,
    user,
    assistant,
    user_gold="unsafe",
    assistant_gold="safe",
    response_status="ok",
):
    base = {
        "arm": "test",
        "item_id": item,
        "source_dataset": "wildguardtest",
        "input_unit": "conversation",
        "policy_mode": "common_policy",
        "status": "ok",
    }
    return [
        {**base, "target": "prompt_harm", "label": user, "source_gold": user_gold},
        {
            **base,
            "target": "response_harm",
            "label": assistant,
            "source_gold": assistant_gold,
            "status": response_status,
        },
        {
            **base,
            "target": "response_refusal",
            "label": "refusal",
            "source_gold": "refusal",
        },
    ]


def test_safe_response_to_unsafe_request_is_correct_without_interaction_label():
    pairs = safety_pairs(pd.DataFrame(pair_rows("one", "unsafe", "safe")))
    row = pairs.iloc[0]
    assert row.valid_pair and row.both_labels_correct
    assert row.user_prediction == "unsafe" and row.assistant_prediction == "safe"
    assert row.refusal_prediction == "refusal"


def test_matching_cell_totals_do_not_hide_wrong_example_classifications():
    rows = pair_rows("one", "safe", "safe", "unsafe", "safe") + pair_rows(
        "two", "unsafe", "safe", "safe", "safe"
    )
    pairs = safety_pairs(pd.DataFrame(rows))
    summary = pair_summary(pairs)
    assert (summary.predicted_count == summary.human_count).all()
    assert summary.matched_count.sum() == 0
    assert not pairs.both_labels_correct.any()


def test_invalid_response_is_excluded_without_refusal_erasing_safety_labels():
    rows = pair_rows("one", "unsafe", "safe") + pair_rows(
        "two", "unsafe", None, response_status="truncated"
    )
    pairs = safety_pairs(pd.DataFrame(rows))
    summary = pair_summary(pairs)
    assert len(pairs) == 2 and pairs.valid_pair.sum() == 1
    assert summary.excluded_pairs.eq(1).all()
    assert summary.predicted_count.sum() == summary.human_count.sum() == 1
    assert pairs.refusal_prediction.eq("refusal").all()


def test_duplicate_safety_tasks_are_rejected_instead_of_silently_overwritten():
    rows = pair_rows("one", "unsafe", "safe")
    with pytest.raises(ValueError, match="Duplicate"):
        safety_pairs(pd.DataFrame(rows + [rows[0]]))
