from run.matrix import cells, option_order, response_companion_cells, companion_cells, matched_option_order


def test_matrix_count_and_determinism():
    plan = list(cells())
    assert len(plan) == 151
    assert len({x["cell_id"] for x in plan}) == 151
    assert len(list(cells(include_optional=True))) == 199
    companion = list(response_companion_cells())
    assert len(companion) == 12
    assert all(x["source"] == "wildguardtest" and x["target"] == "response_harm" for x in companion)
    all_companions = list(companion_cells())
    assert len(all_companions) == 34
    assert {x["target"] for x in all_companions} == {"prompt_harm", "response_harm", "response_refusal"}
    options = ["safe", "unsafe", "needs_caution"]
    assert option_order("item-1", "cell-1", options) == option_order("item-1", "cell-1", options)
    assert set(option_order("item-1", "cell-1", options)) == set(options)


def test_supported_contexts_and_targets_are_explicit():
    plan = list(cells(include_optional=True)) + list(companion_cells(True))
    assert len([c for c in cells() if c["arm"] == "decisions_luna"]) == 16
    assert len([c for c in companion_cells() if c["arm"] == "decisions_luna"]) == 3
    companions = list(companion_cells())
    assert not any(c["arm"] == "jev_noul" for c in companions)
    for cell in companions:
        if cell["arm"] in {"granite_41_8b", "qwen3guard_8b", "nemotron_guard"}:
            assert cell["target"] != "response_refusal"
        expected_context = "user_only" if (cell["arm"], cell["target"]) in {("qwen3guard_8b", "prompt_harm"), ("policylm_17b_prompt", "prompt_harm")} else "full_pair"
        assert cell["input_context"] == expected_context
    granite = [c for c in cells() if c["arm"] == "granite_41_8b"]
    assert len(granite) == 1 and granite[0]["decision_mode"] == "derived_or"


def test_policylm_is_only_a_prompt_reference():
    assert not any(c['arm'] == 'policylm_17b_prompt' for c in cells(True))
    refs = [c for c in companion_cells(True) if c['arm'] == 'policylm_17b_prompt']
    assert len(refs) == 1
    assert refs[0]['target'] == 'prompt_harm' and refs[0]['input_context'] == 'user_only'


def test_order_is_matched_across_reordered_option_inputs():
    # Label identity, not incidental dict construction order, determines presentation.
    assert matched_option_order("x", "response_harm", "s2", ["safe", "unsafe"]) == matched_option_order("x", "response_harm", "s2", ["unsafe", "safe"])
