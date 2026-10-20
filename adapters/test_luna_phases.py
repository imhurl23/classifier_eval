from types import SimpleNamespace as NS

from adapters.luna import parse_response


def response(messages, text=''):
    return NS(status='completed', output=messages, output_text=text, usage=None)


def message(phase, text):
    return NS(type='message', phase=phase, content=[NS(type='output_text', text=text)])


def test_final_json_is_parsed_without_decoding_commentary():
    r = response([message('commentary', 'Analysis of the request.'),
                  message('final_answer', '{"label":"safe"}')],
                 'Analysis of the request.{"label":"safe"}')
    d = parse_response(r, {}, ['safe', 'unsafe'], 0)
    assert d.status == 'ok' and d.label == 'safe'


def test_commentary_without_final_answer_is_not_a_decision():
    r = response([message('commentary', '{"label":"unsafe"}')], '{"label":"unsafe"}')
    assert parse_response(r, {}, ['safe', 'unsafe'], 0).status == 'invalid_output'


def test_unphased_response_keeps_existing_structured_contract():
    r = response([message(None, '{"label":"unsafe"}')], '{"label":"unsafe"}')
    assert parse_response(r, {}, ['safe', 'unsafe'], 0).label == 'unsafe'


def test_multiple_final_answers_are_ambiguous():
    r = response([message('final_answer', '{"label":"unsafe"}'),
                  message('final_answer', '{"label":"safe"}')])
    assert parse_response(r, {}, ['safe', 'unsafe'], 0).status == 'invalid_output'
