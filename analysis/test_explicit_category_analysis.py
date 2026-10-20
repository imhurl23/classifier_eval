import pandas as pd
import pytest

from analysis.explicit_categories import (
    paired_transition,
    recover_saved_final,
    summarize,
)


def test_paired_changes_count_fixes_and_regressions_on_the_same_examples():
    rows = []
    for item, a, b in [
        ("fixed", True, False),
        ("regressed", False, True),
        ("same", True, True),
    ]:
        rows += [
            {
                "item_id": item,
                "variant": "original_policy_repeat",
                "source_gold": "unsafe",
                "error": a,
            },
            {
                "item_id": item,
                "variant": "explicit_categories",
                "source_gold": "unsafe",
                "error": b,
            },
        ]
    result = paired_transition(pd.DataFrame(rows))
    assert result["paired_n"] == 3
    assert result["corrected"] == result["introduced"] == result["both_wrong"] == 1
    assert result["both_correct"] == 0 and result["delta_pp"] == 0


def test_refusal_and_safe_prediction_are_both_strict_misses_but_counted_separately():
    g = pd.DataFrame(
        [
            {"source_gold": "unsafe", "status": "refusal", "label": None},
            {"source_gold": "unsafe", "status": "ok", "label": "safe"},
            {"source_gold": "unsafe", "status": "ok", "label": "unsafe"},
        ]
    )
    result = summarize(g)
    assert result["n"] == 3 and result["errors"] == 2
    assert result["invalid"] == result["wrong_label"] == 1


def test_duplicate_condition_is_rejected_instead_of_inflating_paired_sample():
    r = {
        "item_id": "one",
        "variant": "explicit_categories",
        "source_gold": "safe",
        "error": False,
    }
    with pytest.raises(AssertionError):
        paired_transition(pd.DataFrame([r, r]))


def test_classifier_refusal_on_safe_prompt_is_not_a_false_alarm():
    g = pd.DataFrame(
        [
            {"source_gold": "safe", "status": "refusal", "label": None},
            {"source_gold": "safe", "status": "ok", "label": "unsafe"},
            {"source_gold": "safe", "status": "ok", "label": "safe"},
        ]
    )
    result = summarize(g)
    assert result["errors"] == 1 and result["strict_errors"] == 2
    assert result["invalid"] == 1


def test_saved_parser_recovery_leaves_raw_evidence_untouched():
    # A minimal saved SDK response retains its phases; replay uses the same parser
    # as live calls without making another request.
    from copy import deepcopy

    row = {
        "arm": "luna_structured",
        "status": "invalid_output",
        "label": None,
        "job_id": "job",
        "item_id": "item",
        "variant": "explicit_categories",
        "latency_ms": 1,
        "raw": {
            "request": {},
            "response": {
                "id": "resp_saved",
                "created_at": 0,
                "model": "gpt-6-luna",
                "object": "response",
                "parallel_tool_calls": True,
                "status": "completed",
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "msg_one",
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "phase": "commentary",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Reasoning text.",
                                "annotations": [],
                            }
                        ],
                    },
                    {
                        "id": "msg_two",
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "phase": "final_answer",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"label":"safe"}',
                                "annotations": [],
                            }
                        ],
                    },
                ],
            },
        },
    }
    frozen = deepcopy(row)
    result, recovery = recover_saved_final(row)
    assert row == frozen
    assert result["status"] == "ok" and result["label"] == "safe"
    assert (
        result["raw"] == row["raw"] and recovery["original_status"] == "invalid_output"
    )
