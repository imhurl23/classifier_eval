from adapters.jev import JevChoiceAdapter, JevNoulAdapter, load_typesafe_key


def test_jev_key_comes_from_repo_env():
    assert load_typesafe_key()


def test_jev_adapter_choice_contract():
    class Response:
        ok = True
        status_code = 200

        def json(self):
            return {"model": "jev-1.13.0", "answers": {"safety": {
                "type": "choice", "choice": "unsafe",
                "probabilities": {"safe": 0.1, "unsafe": 0.9}, "confidence": 0.8}},
                "usage": {"input_tokens": 20, "output_tokens": 3}}

    class Session:
        def post(self, url, headers, json, timeout):
            assert url.endswith("/v1/systemone")
            assert headers["Authorization"].startswith("Bearer ")
            assert list(json["questions"]["safety"]["criteria"]) == ["unsafe", "safe"]
            return Response()

    adapter = JevChoiceAdapter({"safe": "Benign", "unsafe": "Harmful"}, session=Session())
    decision = adapter.classify("sample", "s2", "rubric", ["unsafe", "safe"])
    assert decision.status == "ok"
    assert decision.label == "unsafe"
    assert decision.probs == {"unsafe": 0.9, "safe": 0.1}
    assert decision.usage["input_tokens"] == 20


def test_jev_can_make_separate_decisions_in_one_request():
    class Response:
        ok = True
        status_code = 200
        def json(self):
            return {"answers": {
                "prompt_harm": {"choice": "unsafe", "probabilities": {"safe": 0.1, "unsafe": 0.9}},
                "response_harm": {"choice": "safe", "probabilities": {"safe": 0.8, "unsafe": 0.2}},
            }, "usage": {"input_tokens": 30}}
    class Session:
        def post(self, url, headers, json, timeout):
            assert list(json["questions"]) == ["prompt_harm", "response_harm"]
            assert json["state"] == "User:\nrequest\n\nAssistant:\nrefusal"
            return Response()
    questions = {name: {"instructions": name, "criteria": {"safe": "safe", "unsafe": "unsafe"}}
                 for name in ("prompt_harm", "response_harm")}
    result = JevChoiceAdapter({"safe": "safe", "unsafe": "unsafe"}, session=Session()).classify_many(
        "User:\nrequest\n\nAssistant:\nrefusal", questions)
    assert result["prompt_harm"].label == "unsafe"
    assert result["response_harm"].label == "safe"


def test_jev_noul_returns_binary_probability():
    class Response:
        ok = True
        status_code = 200
        def json(self):
            return {"answers": {"unsafe": {"type": "noul", "noul": 0.8}},
                    "usage": {"input_tokens": 10}}
    class Session:
        def post(self, url, headers, json, timeout):
            assert json["questions"]["unsafe"]["type"] == "noul"
            assert json["questions"]["unsafe"]["criteria"]["true"] == "harmful"
            return Response()
    result = JevNoulAdapter({"safe": "safe", "unsafe": "harmful"}, Session()).classify(
        "User:\ntext", "s2", "rubric", ["safe", "unsafe"])
    assert result.label == "unsafe" and result.probs == {"safe": 0.19999999999999996, "unsafe": 0.8}
