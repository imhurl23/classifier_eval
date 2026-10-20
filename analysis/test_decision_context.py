import pandas as pd
import pytest

from analysis.decision_context import collapse, matched


def rows(labels, gold=None, status=None):
    return pd.DataFrame(
        dict(
            item_id=[str(i) for i in range(len(labels))],
            label=labels,
            source_gold=gold or ["unsafe"] * len(labels),
            status=status or ["ok"] * len(labels),
        )
    )


def test_category_caution_and_failure_are_distinct():
    r = rows(
        ["violence", "safe", "needs_caution", None],
        status=["ok", "ok", "ok", "refusal"],
    )
    assert collapse(r).tolist() == ["unsafe", "safe", "needs_caution", "no_verdict"]


def test_net_accuracy_can_hide_changed_decisions():
    summary, cases = matched(rows(["unsafe", "safe"]), rows(["safe", "violence"]))
    assert summary["changed"] == 2 and summary["gained"] == summary["lost"] == 1
    assert summary["base_correct"] == summary["variant_correct"] == 1


def test_matched_diagnostics_reject_duplicates_and_missing_cases():
    base = rows(["safe", "unsafe"])
    with pytest.raises(AssertionError):
        matched(base, base.iloc[:1])
    with pytest.raises(AssertionError):
        matched(base, pd.concat([base, base.iloc[:1]]))


def test_consistent_wrong_decisions_remain_visible():
    summary, _ = matched(rows(["safe", "unsafe"]), rows(["safe", "unsafe"]))
    assert summary["changed"] == 0 and summary["both_wrong"] == 1
