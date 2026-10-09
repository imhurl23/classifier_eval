import yaml
from pathlib import Path
from training.generate_constitutional import assign_responses, expand_records, QUESTIONS, RESPONSE_SHARE

ROOT = Path(__file__).resolve().parents[1]


def _items():
    base = {"language": "English", "style": "restricted_vanilla", "gen_model": "m", "k": 0}
    return [dict(base, text=f"restricted {i}", kind="restricted", **{"class": "harassment"}) for i in range(60)] + \
           [dict(base, text=f"benign {i}", kind="safe", style="permitted_nearmiss", **{"class": "harassment"}) for i in range(60)] + \
           [dict(base, text=f"ambiguous {i}", kind="needs_caution", style="needs_caution", **{"class": "violence"}) for i in range(30)]


def test_response_assignment_is_deterministic_and_near_target_share():
    a = assign_responses(_items()); b = assign_responses(_items())
    assert [x["needs_response"] for x in a] == [x["needs_response"] for x in b]
    share = sum(x["needs_response"] for x in a) / len(a)
    assert abs(share - RESPONSE_SHARE) < 0.15
    assert all(x["response_style"] != "harmful_response" for x in a if x["kind"] != "safe")


def test_interaction_labels_follow_policy():
    ontology = yaml.safe_load((ROOT / "data/label_map.yaml").read_text())
    defs = yaml.safe_load((ROOT / "prompts/label_definitions.yaml").read_text())
    items = assign_responses(_items())
    for it in items:
        if it["needs_response"]:
            it["response"] = "Sure, here is a reply."
    recs = expand_records(items, ontology, defs, QUESTIONS)
    by = {}
    for r in recs:
        by.setdefault(r["id"].split(":")[0], {})[r["scheme"]] = r
    # restricted user turn with a refusal-style reply stays unsafe; prompt-only stays unsafe
    for i in range(60):
        r = by[f"cc-{i:06d}"]
        assert r["s2"]["answer_key"] == "unsafe" and r["s13"]["answer_key"] == "harassment"
        state = r["s2"]["state"]
        assert state.startswith("User:\n") and (("Assistant:" in state) == bool(items[i].get("response")))
    # benign user turn: safe unless the reply is the harmful one
    for i in range(60, 120):
        it, r = items[i], by[f"cc-{i:06d}"]
        if it.get("response") and it["response_style"] == "harmful_response":
            assert r["s2"]["answer_key"] == "unsafe" and r["s13"]["answer_key"] == it["response_class"]
        else:
            assert r["s2"]["answer_key"] == "safe" and r["s13"]["answer_key"] == "safe"
    # needs_caution has no s2 record
    for i in range(120, 150):
        r = by[f"cc-{i:06d}"]
        assert "s2" not in r and r["s3"]["answer_key"] == "needs_caution"
