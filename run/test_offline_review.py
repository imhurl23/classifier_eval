import json

from data.build_datasets import build, normalize_text
from prompts.runtime import instructions_for
from run.offline_review import (CaptureGuard, assert_context, bundled_request, jobs,
                                prepare_sample, request_for)


def test_sample_is_blinded_unique_and_disjoint_from_held_out():
    datasets, _, _ = build()
    rows, blind, private, info = prepare_sample(datasets)
    held = {normalize_text(r['input']['prompt']) for r in datasets['dis-test'] + datasets['dis-pilot']}
    assert len(rows) == len(blind) == 200
    assert len({normalize_text(r['input']['prompt']) for r in rows}) == 200
    assert not held & {normalize_text(r['input']['prompt']) for r in rows}
    assert all(r['input']['unit'] == 'conversation' for r in rows)
    assert all(set(b) == {'audit_id', 'user', 'assistant'} for b in blind)
    assert [p['audit_id'] for p in private] == [b['audit_id'] for b in blind]
    assert all(r['metadata']['split'] == 'validation' for r in rows)
    assert prepare_sample(datasets)[1] == blind


def test_every_planned_request_preserves_its_declared_context():
    row = {'id':'synthetic', 'input':{'text':'User:\nHello\n\nAssistant:\nHi',
                                    'prompt':'Hello', 'response':'Hi'}}
    for _, arm, r, target, rubric, scheme, _ in jobs([row]):
        request = bundled_request(arm,r) if target == 'bundle' else request_for(arm,r,target,rubric,scheme)
        assert_context(arm,r,target,request)
        assert 'Authorization' not in json.dumps(request)
        if arm == 'granite_41_8b':
            criterion = request['messages'][-1]['content']
            assert 'privacy invasion' in criterion and 'past wrongdoing' in criterion
        if arm in {'jev_choice','clef_choice','clef_flash_choice'} and target == 'prompt_harm':
            bundle = bundled_request(arm,r)
            assert list(request['questions']['safety']['criteria']) == list(bundle['questions']['prompt_harm']['criteria'])


def test_process_control_does_not_override_adapter_output_format():
    text = instructions_for('r_neutral_long').lower()
    for phrase in ('do not provide a confidence', 'do not wrap', 'copied exactly', 'no additional fields', 'return the identifier'):
        assert phrase not in text
    assert 'calling interface' in text
