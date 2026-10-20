import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from prompts.source_aligned.runtime import policy_for
from run import source_policy_revision as runner


@pytest.fixture(scope='module')
def rows():
    return {r['id']: r for r in runner.records(runner.SOURCE / 'sample.jsonl')}


def test_only_independent_human_tasks(rows):
    for row in rows.values():
        for target in runner.tasks(row):
            assert not (row['metadata']['source'].startswith('aegis') and target != 'interaction_harm')
            assert not (row['metadata']['source'] == 'wildguardtest' and target == 'interaction_harm')
    assert runner.tasks(next(r for r in rows.values() if r['metadata']['source'] == 'xstest')) == ()


def test_boundaries_and_option_definitions_agree():
    policy, defs = policy_for('aegis_v2_test', 'interaction_harm', scheme='s13')
    assert 'untargeted' in policy and 'untargeted' in defs['profanity']
    assert 'completed' in policy and 'completed' in defs['criminal_planning_confessions']
    assert 'confessions' in defs['suicide_self_harm']
    assert 'Needs Caution into Safe' in policy
    policy, defs = policy_for('wildguardtest', 'response_refusal')
    assert 'initially saying no but then answering' in policy
    assert 'refusal then compliance' in defs['not_refusal']
    with pytest.raises(ValueError):
        policy_for('aegis_v2_test', 'response_harm')


@pytest.mark.parametrize('arm', runner.ARMS)
def test_payload_contains_revised_policy_and_exact_original_input(arm, rows):
    for item, target in [('aegis-validation-000415', 'interaction_harm'), ('wildguard-000425', 'response_refusal')]:
        if arm == 'jev_noul' and target == 'response_refusal':
            continue
        row = rows[item]; job = runner.job_for(arm, row, target)
        wire = json.dumps(job['request'], ensure_ascii=False)
        def strings(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, list):
                return [s for v in value for s in strings(v)]
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except (ValueError, TypeError):
                    return [value]
                return [value] + (strings(decoded) if isinstance(decoded, (dict, list)) else [])
            return []
        assert any(row['input']['text'] in s for s in strings(job['request']))
        assert job['request_sha256'] == runner.digest(job['request'])
        assert item not in wire
        assert job['source_gold'] not in {None, ''}


def test_neutral_length_and_withheld_information():
    tok = Tokenizer.from_file(str(runner.ROOT / 'data/raw/qwen-tokenizer/tokenizer.json'))
    for scheme in ('s2', 's3', 's6', 's13'):
        long, defs = policy_for('aegis_v2_validation', 'interaction_harm', 'long', scheme)
        neutral, other = policy_for('aegis_v2_validation', 'interaction_harm', 'neutral_long', scheme)
        a = len(tok.encode(long).ids); b = len(tok.encode(neutral).ids)
        assert abs(a-b) / a < .03
        assert defs == other
        withheld, defs = policy_for('aegis_v2_validation', 'interaction_harm', 'withheld', scheme)
        assert 'profanity' not in withheld.lower()
        assert all(k == v for k, v in defs.items())


def test_paid_budget_reservations_and_unknown_usage_are_not_free(tmp_path, monkeypatch):
    from adapters.base import Decision
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'HOSTED_CAP', .01)
    job = dict(arm='claude_structured', job_id='one', input_tokens_proxy=100, request={'max_tokens': 256})
    runner.budget_update(job)
    with pytest.raises(RuntimeError, match='budget'):
        runner.budget_update({**job, 'job_id': 'two'})
    runner.budget_update(job, Decision(None, None, {}, 'api_error', 0, {}))
    ledger = json.loads((tmp_path / 'hosted-budget.json').read_text())
    assert ledger['charged_estimate_usd'] > 0 and not ledger['reserved']
