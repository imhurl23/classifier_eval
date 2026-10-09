from adapters.clef import make_choice_request, parse_choice, ClefChoiceAdapter


def test_choice_request_keeps_option_order_and_rubric():
    request = make_choice_request("sample", "Use the rubric", ["unsafe", "safe"],
                                  {"safe": "Permitted", "unsafe": "Disallowed"}, "clef")
    assert list(request["questions"]["safety"]["criteria"]) == ["unsafe", "safe"]
    assert request["questions"]["safety"]["instructions"] == "Use the rubric"


def test_parse_wrapped_response_and_reject_missing_probability():
    request = {}
    good = {"success": True, "result": {"answers": {"safety": {
        "choice": "unsafe", "probabilities": {"safe": 0.2, "unsafe": 0.8}}},
        "usage": {"input_tokens": 12}}}
    decision = parse_choice(good, ["unsafe", "safe"], request, 5)
    assert decision.status == "ok"
    assert decision.probs == {"unsafe": 0.8, "safe": 0.2}
    bad = {"answers": {"safety": {"choice": "unsafe", "probabilities": {"unsafe": 1.0}}}}
    assert parse_choice(bad, ["unsafe", "safe"], request, 5).status == "invalid_output"


def test_adapter_requires_cloudflare_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AUTH_TOKEN", raising=False)
    adapter = ClefChoiceAdapter("clef_choice", {"safe": "Permitted", "unsafe": "Disallowed"})
    try:
        adapter.classify("sample", "s2", "rubric", ["safe", "unsafe"])
    except RuntimeError as exc:
        assert "Cloudflare" in str(exc)
    else:
        raise AssertionError("Expected missing credentials to block call")


def test_adapter_live_path_with_fake_transport():
    class Response:
        status_code = 200
        ok = True

        def json(self):
            return {"success": True, "result": {"answers": {"safety": {
                "choice": "safe", "probabilities": {"safe": 0.9, "unsafe": 0.1}}}}}

    class Session:
        def post(self, url, headers, json, timeout):
            assert url.endswith("/ai/run/@cf/cloudflare/clef")
            assert headers["Authorization"] == "Bearer token"
            assert list(json["questions"]["safety"]["criteria"]) == ["unsafe", "safe"]
            assert timeout == 60
            return Response()

    adapter = ClefChoiceAdapter("clef_choice", {"safe": "Permitted", "unsafe": "Disallowed"},
                                account_id="acct", auth_token="token", session=Session())
    result = adapter.classify("sample", "s2", "rubric", ["unsafe", "safe"])
    assert result.status == "ok"
    assert result.label == "safe"
    assert "token" not in str(result.raw)
