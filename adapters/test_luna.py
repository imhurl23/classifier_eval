import json
from types import SimpleNamespace as NS

from adapters.luna import LunaStructuredAdapter, make_request


def test_luna_request_is_structured_and_uses_one_state():
    req = make_request("User:\nhello", "Classify safety", ["unsafe", "safe"],
                       {"safe": "benign", "unsafe": "harmful"})
    assert req["reasoning"] == {"effort": "none"}
    assert req["text"]["format"]["schema"]["properties"]["label"]["enum"] == ["unsafe", "safe"]
    assert req["input"][1]["content"] == "User:\nhello"


def test_luna_adapter_parses_response():
    class Responses:
        def create(self, **request):
            return NS(status="completed", model="gpt-6-luna", output_text=json.dumps({"label": "safe"}),
                      output=[], usage=NS(model_dump=lambda: {"input_tokens": 10}),
                      model_dump=lambda: {"output_text": '{"label":"safe"}'})
    adapter = LunaStructuredAdapter({"safe": "benign", "unsafe": "harmful"},
                                     client=NS(responses=Responses()))
    result = adapter.classify("User:\nhello", "s2", "Classify", ["safe", "unsafe"])
    assert result.status == "ok" and result.label == "safe" and result.probs is None
