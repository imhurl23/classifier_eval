from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import requests

from adapters.openai_decisions import (API_URL, OpenAIDecisionsAdapter,
    make_choice_request, make_multi_choice_request, parse_multi_choice)
from run.pilot import SUPPORTED, adapter_for

DEFINITIONS = {'safe': 'benign', 'unsafe': 'harmful'}


def request():
    return make_choice_request('User:\nhello\n\nAssistant:\nhi', 'Classify safety',
                               ['unsafe', 'safe'], DEFINITIONS)


def payload():
    return {'model': 'gpt-6-luna', 'answers': [{'name': 'safety', 'type': 'choice',
            'choice': 'safe', 'confidence': .9,
            'probabilities': [{'value': 'safe', 'probability': .9},
                              {'value': 'unsafe', 'probability': .1}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 0}}


def test_direct_adapter_factory_and_ordered_request(monkeypatch):
    monkeypatch.setattr('run.pilot.load_local_service_keys', lambda: None)
    assert 'decisions_luna' in SUPPORTED
    assert isinstance(adapter_for('decisions_luna', DEFINITIONS), OpenAIDecisionsAdapter)
    req = request()
    assert req['input'].count('User:') == 1
    assert [c['value'] for c in req['questions'][0]['choices']] == ['unsafe', 'safe']
    with pytest.raises(ValueError):
        make_choice_request('x', 'y', ['safe', 'safe'], DEFINITIONS)


def test_choice_preserves_distribution_and_usage():
    result = parse_multi_choice(payload(), request(), 12)['safety']
    assert result.status == 'ok' and result.label == 'safe'
    assert result.probs == {'safe': .9, 'unsafe': .1}
    assert result.usage['input_tokens'] == 10


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'sum', 'nan', 'unknown', 'wrong_name', 'wrong_type', 'extra_answer', 'null'])
def test_malformed_answers_are_not_scored(change):
    body = payload()
    answer = body['answers'][0]
    if change == 'missing':
        answer['probabilities'].pop()
    elif change == 'duplicate':
        answer['probabilities'][1]['value'] = 'safe'
    elif change == 'sum':
        answer['probabilities'][0]['probability'] = .5
    elif change == 'nan':
        answer['probabilities'][0]['probability'] = float('nan')
    elif change == 'unknown':
        answer['choice'] = 'other'
    elif change == 'wrong_name':
        answer['name'] = 'other'
    elif change == 'wrong_type':
        answer['type'] = 'score'
    elif change == 'extra_answer':
        body['answers'].append(deepcopy(answer))
    else:
        body = None
    result = parse_multi_choice(body, request(), 0)['safety']
    assert result.status == 'invalid_output' and result.label is None and result.probs is None


def test_batched_refusal_is_distinct_from_safety_label():
    req = make_multi_choice_request('pair', {
        name: {'instructions': name, 'criteria': DEFINITIONS} for name in ['harm', 'refusal']})
    body = payload()
    body['answers'][0]['name'] = 'harm'
    body['answers'].append({'type': 'refusal', 'name': 'refusal'})
    results = parse_multi_choice(body, req, 0)
    assert results['harm'].status == 'ok'
    assert results['refusal'].status == 'refusal' and results['refusal'].label is None


@pytest.mark.parametrize('failure', ['http', 'non_json', 'network', 'retry'])
def test_transport_errors_and_retry(monkeypatch, failure):
    monkeypatch.setattr('adapters.openai_decisions.load_openai_key', lambda: 'secret')
    monkeypatch.setattr('adapters.openai_decisions.time.sleep', lambda _: None)
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        assert url == API_URL and kwargs['headers']['Authorization'] == 'Bearer secret'
        if failure == 'network':
            raise requests.ConnectionError('secret')
        def decode():
            if failure == 'non_json':
                raise ValueError('bad json')
            return payload()
        retrying = failure == 'retry' and len(calls) == 1
        return NS(status_code=429 if retrying else 401 if failure == 'http' else 200,
                  ok=not retrying and failure != 'http', json=decode, text='not-json')
    adapter = OpenAIDecisionsAdapter(DEFINITIONS, session=NS(post=post))
    result = adapter.classify('pair', 's2', 'Classify', ['safe', 'unsafe'])
    assert result.status == ('ok' if failure == 'retry' else 'invalid_output' if failure == 'non_json' else 'api_error')
    assert 'secret' not in str(result.raw)
    assert len(calls) == (2 if failure == 'retry' else 1)
