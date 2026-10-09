import json
from types import SimpleNamespace as NS

from adapters.claude import ClaudeStructuredAdapter, GATEWAY_BASE_URL, make_request, parse_response

DEFS = {"safe": "Permitted", "unsafe": "Disallowed"}


def _resp(text=None, stop="end_turn", model="claude-sonnet-5-5"):
    content = [NS(type="text", text=text)] if text is not None else []
    return NS(content=content, stop_reason=stop, stop_details=NS(category="general_harms") if stop == "refusal" else None,
              model=model, usage=NS(input_tokens=200, output_tokens=8), to_dict=lambda: {"stop_reason": stop})


def test_request_has_enum_schema_low_effort_and_ordered_definitions():
    req = make_request("hello", "Rubric here", ["unsafe", "safe"], DEFS)
    assert req["model"] == "claude-sonnet-5-5"
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == ["unsafe", "safe"]
    assert req["system"].index("- unsafe:") < req["system"].index("- safe:")
    assert req["messages"] == [{"role": "user", "content": "hello"}]
    assert "fallbacks" not in req and "thinking" not in req


def test_parse_ok_refusal_truncated_invalid():
    ok = parse_response(_resp(json.dumps({"label": "safe"})), ["unsafe", "safe"], {}, 3)
    assert ok.status == "ok" and ok.label == "safe" and ok.probs is None
    assert ok.raw["served_model"] == "claude-sonnet-5-5" and ok.usage["input_tokens"] == 200
    ref = parse_response(_resp(None, stop="refusal"), ["unsafe", "safe"], {}, 3)
    assert ref.status == "refusal" and ref.raw["refusal_category"] == "general_harms"
    assert parse_response(_resp("{", stop="max_tokens"), ["unsafe", "safe"], {}, 3).status == "truncated"
    assert parse_response(_resp(json.dumps({"label": "maybe"})), ["unsafe", "safe"], {}, 3).status == "invalid_output"


def test_adapter_uses_injected_client_and_pins_served_model():
    calls = []
    class Messages:
        def create(self, **kw):
            calls.append(kw); return _resp(json.dumps({"label": "unsafe"}), model="claude-sonnet-5-5")
    adapter = ClaudeStructuredAdapter(DEFS, client=NS(messages=Messages()))
    d = adapter.classify("text", "s2", "rubric", ["safe", "unsafe"])
    assert d.status == "ok" and d.label == "unsafe"
    assert adapter.model_version == "claude-sonnet-5-5"
    assert calls[0]["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == ["safe", "unsafe"]
    assert GATEWAY_BASE_URL.endswith("/proxy/anthropic")


def test_adapter_requires_braintrust_key(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    try:
        ClaudeStructuredAdapter(DEFS)
    except RuntimeError as exc:
        assert "BRAINTRUST_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing key to fail")
