from prompts.runtime import instructions_for, options_for
from data.build_datasets import PILOT_WG_IDS, build


def test_runtime_instructions_send_state_once_and_no_placeholders():
    for name in ("r_short", "r_long", "r_neutral_long", "r_withheld",
                 "prompt_harm_s2", "response_harm_s2", "response_refusal_s2"):
        instruction = instructions_for(name)
        assert "{text}" not in instruction
        assert "{allowed_labels}" not in instruction
        assert "{label_definitions}" not in instruction
        assert "Interaction:\n" not in instruction


def test_withheld_has_no_policy_definitions_in_choice_criteria():
    labels, criteria = options_for("s13", "r_withheld")
    assert criteria == {label: label for label in labels}
    _, ordinary = options_for("s13", "r_short")
    assert ordinary["safe"] != "safe"


def test_turn_specific_definitions_do_not_use_interaction_or():
    _, prompt = options_for("s2", "prompt_harm_s2")
    _, response = options_for("s2", "response_harm_s2")
    assert "User turn" in prompt["safe"] and "Assistant turn" not in prompt["safe"]
    assert "Assistant turn" in response["safe"] and "User turn" not in response["safe"]


def test_pilot_interactions_are_excluded_from_test_and_controls():
    datasets, _, _ = build()
    assert {r["id"] for r in datasets["dis-pilot"]} == PILOT_WG_IDS
    held_out = {r["id"] for name in ("dis-test", "dis-controls-200")
                for r in datasets[name]}
    assert not PILOT_WG_IDS & held_out


def test_long_examples_do_not_repeat_previous_dev_items():
    long = instructions_for("r_long").lower()
    for phrase in ("formal essay about kidnapping", "hateful thing to say to a woman",
                   "what bones are the easiest to break", "puncture someone's tires"):
        assert phrase not in long


def test_all_harm_targets_state_the_covered_harms():
    for rubric in ("r_short", "r_long", "r_neutral_long", "prompt_harm_s2", "response_harm_s2"):
        instruction = instructions_for(rubric)
        for covered in ("privacy invasion", "self-harm encouragement", "fraud", "malware"):
            assert covered in instruction


def test_refusal_options_allow_mixed_refusal_and_assistance():
    _, criteria = options_for("s2", "response_refusal_s2")
    assert "all or part" in criteria["refusal"]
    assert "both a refusal and harmful assistance" in instructions_for("response_refusal_s2")
