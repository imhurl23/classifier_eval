import json
import math

from adapters.tev1 import LettersAdapter, make_request, parse_response, SYSTEM

DEFS = {"safe": "Permitted", "unsafe": "Disallowed", "needs_caution": "Unclear"}
ORDER = ["unsafe", "safe", "needs_caution"]


def _payload(content, top, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish,
                         "logprobs": {"content": [{"token": content.strip(),
                                                   "top_logprobs": [{"token": t, "logprob": lp}
                                                                    for t, lp in top]}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 1}}


def test_request_follows_tev1_contract():
    req = make_request("hello", "Classify it", ORDER, DEFS, "m")
    assert req["messages"][0]["content"] == SYSTEM
    body = json.loads(req["messages"][1]["content"])
    assert body["state"] == "hello" and body["question"] == "Classify it"
    assert [o["label"] for o in body["options"]] == ["A", "B", "C"]
    assert [o["key"] for o in body["options"]] == ORDER
    assert req["temperature"] == 0 and req["max_tokens"] == 8
    assert req["logprobs"] is True and req["top_logprobs"] == 20
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in req


def test_letter_maps_to_option_and_probs_renormalise():
    d = parse_response(_payload("A", [("A", math.log(0.6)), ("B", math.log(0.3)),
                                      ("C", math.log(0.05)), ("\n", math.log(0.05))]),
                       ORDER, {}, 5)
    assert d.status == "ok" and d.label == "unsafe"
    assert abs(sum(d.probs.values()) - 1) < 1e-9
    assert abs(d.probs["unsafe"] - 0.6 / 0.95) < 1e-9
    assert abs(d.raw["letter_mass"] - 0.95) < 1e-9


def test_missing_letter_in_top_logprobs_is_scoring_miss_but_keeps_label():
    d = parse_response(_payload("B", [("B", -0.1), ("A", -3.0)]), ORDER, {}, 5)
    assert d.status == "ok" and d.label == "safe"
    assert d.probs is None and d.raw["scoring_miss"] == ["C"]


def test_merges_whitespace_token_variants():
    d = parse_response(_payload("A", [("A", math.log(0.3)), (" A", math.log(0.3)),
                                      ("B", math.log(0.2)), ("C", math.log(0.2))]),
                       ORDER, {}, 5)
    assert abs(d.probs["unsafe"] - 0.6) < 1e-9


def test_prose_or_unlisted_letter_is_invalid():
    assert parse_response(_payload("A because", [("A", -0.1)]), ORDER, {}, 5).status == "invalid_output"
    assert parse_response(_payload("D", [("D", -0.1)]), ORDER, {}, 5).status == "invalid_output"
    assert parse_response({"error": "boom"}, ORDER, {}, 5).status == "invalid_output"


def test_adapter_requires_credentials(monkeypatch):
    for var in ("BASETEN_API_KEY", "BASETEN_TEV1_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    adapter = LettersAdapter("tev1_native", DEFS)
    try:
        adapter.classify("x", "s3", "rubric", ORDER)
    except RuntimeError as exc:
        assert "baseten" in str(exc)
    else:
        raise AssertionError("Expected missing credentials to block call")


def test_live_path_with_fake_transport_baseten_and_together():
    class Response:
        status_code, ok = 200, True
        def json(self):
            return _payload("C", [("C", -0.05), ("A", -4.0), ("B", -4.0)])

    class Session:
        def __init__(self): self.calls = []
        def post(self, url, headers, json, timeout):
            self.calls.append((url, headers["Authorization"], json["model"]))
            return Response()

    s = Session()
    a = LettersAdapter("tev1_native", DEFS, provider="baseten", api_key="k",
                       baseten_model_id="abc123", session=s)
    d = a.classify("x", "s3", "rubric", ORDER)
    assert d.status == "ok" and d.label == "needs_caution"
    assert s.calls[0] == ("https://model-abc123.api.baseten.co/environments/production/sync/v1/chat/completions",
                          "Api-Key k", "togethercomputer/Tev1-4B-experimental")
    assert "Api-Key" not in str(d.raw) and " k" not in str(d.raw["request"])

    s = Session()
    b = LettersAdapter("qwen35_4b_letters", DEFS, provider="together", api_key="t", session=s)
    b.classify("x", "s3", "rubric", ORDER)
    assert s.calls[0] == ("https://api.together.ai/v1/chat/completions", "Bearer t", "Qwen/Qwen3.5-4B")
    assert a.model_version.startswith("hf:0b7becf0") and b.model_version.startswith("hf:851bf6e8")
