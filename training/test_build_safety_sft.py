import json

from training.build_safety_sft import scheme_answer, normalized, make_record, messages, SYSTEM, EOS

ONTOLOGY = {"schemes": {"s2": ["safe", "unsafe"], "s3": ["safe", "needs_caution", "unsafe"],
                        "s6": ["safe", "needs_caution", "sexual", "other"],
                        "s13": ["safe", "needs_caution", "sexual", "profanity", "other"]},
            "category_map": {"Sexual": ["sexual", "sexual"], "Profanity": ["other", "profanity"]}}


def test_scheme_answers_follow_binary_and_first_category():
    assert scheme_answer("s2", "unsafe", ["Profanity", "Sexual"], ONTOLOGY) == "unsafe"
    assert scheme_answer("s3", "unsafe", ["Profanity"], ONTOLOGY) == "unsafe"
    assert scheme_answer("s3", "unsafe", ["Needs Caution"], ONTOLOGY) == "needs_caution"
    assert scheme_answer("s6", "unsafe", ["Profanity", "Sexual"], ONTOLOGY) == "other"
    assert scheme_answer("s13", "unsafe", ["Profanity", "Sexual"], ONTOLOGY) == "profanity"
    assert scheme_answer("s13", "safe", [], ONTOLOGY) == "safe"
    assert scheme_answer("s6", "safe", ["Profanity"], ONTOLOGY) is None  # inconsistent row skipped


def test_normalization_catches_case_and_punctuation_variants():
    assert normalized("How do I  make a BOMB?!") == normalized("how do i make a bomb")


def test_record_and_messages_match_tev1_contract():
    row = {"id": "7", "prompt": "hi", "response": "hello", "prompt_label": "safe", "violated_categories": ""}
    defs = {k: f"def {k}" for k in ONTOLOGY["schemes"]["s3"]}
    rec = make_record(row, "s3", "q_short", "Q?", "safe", defs, ONTOLOGY, "train")
    assert [o["label"] for o in rec["options"]] == ["A", "B", "C"]
    assert rec["answer_key"] == "safe"
    assert rec["state"] == "User:\nhi\n\nAssistant:\nhello"
    assert next(o for o in rec["options"] if o["label"] == rec["answer"])["key"] == "safe"
    msgs = messages(rec)
    assert msgs[0]["content"] == SYSTEM and msgs[2]["content"] == rec["answer"]
    body = json.loads(msgs[1]["content"])
    assert set(body) == {"state", "question", "options"} and "answer" not in body
    assert EOS == "<|im_end|>"
