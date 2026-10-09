import json
from pathlib import Path

import pytest

from run.development import load_plan, reserve_usd, run, scores_for
from run.offline_review import digest, request_for


@pytest.fixture(autouse=True)
def isolated_revision(monkeypatch,tmp_path):
    monkeypatch.setattr('run.development.ROOT',tmp_path)


def make_plan(path):
    row={'id':'x','input':{'text':'User:\nHello\n\nAssistant:\nHi','prompt':'Hello','response':'Hi','unit':'conversation'},'metadata':{'split':'validation'}}
    request=request_for('decisions_luna',row,'interaction_harm','r_long','s2')
    job={'arm':'decisions_luna','item_id':'x','target':'interaction_harm','rubric':'r_long','scheme':'s2','replicate':0,'request':request,'request_sha256':digest(request)}
    (path/'summary.json').write_text(json.dumps({'version':'test','file_sha256':{},'requests':1,'items':1}))
    (path/'sample.jsonl').write_text(json.dumps(row)+'\n')
    (path/'manifest.jsonl').write_text(json.dumps(job)+'\n')
    return row,job


def test_frozen_request_and_development_split_are_required(tmp_path):
    row,job=make_plan(tmp_path)
    assert len(load_plan(tmp_path)[2])==1
    job['request']['input']='different evidence'
    (tmp_path/'manifest.jsonl').write_text(json.dumps(job)+'\n')
    with pytest.raises(ValueError,match='Frozen request'):
        load_plan(tmp_path)
    row,job=make_plan(tmp_path)
    row['metadata']['split']='test'
    (tmp_path/'sample.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='development pairs'):
        load_plan(tmp_path)


def test_budget_check_happens_before_braintrust_or_paid_calls(tmp_path):
    _,job=make_plan(tmp_path)
    assert reserve_usd(job)>0
    with pytest.raises(ValueError,match='exceeds budget'):
        run(tmp_path,0.000001)
    assert not (tmp_path/'execution.json').exists()


def test_runner_logs_through_span_and_checkpoints_before_logging(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import braintrust
    from adapters.base import Decision
    make_plan(tmp_path)
    logged=[]
    class Span:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def log(self,**event):
            assert (tmp_path/'results.jsonl').exists()
            logged.append(event)
    experiment=NS(summarize=lambda:NS(experiment_url='test'),start_span=lambda **_:Span(),flush=lambda:None)
    monkeypatch.setattr(braintrust,'init',lambda **_:experiment)
    monkeypatch.setattr('run.development.load_braintrust_key',lambda:None)
    adapter=NS(model_version='test',classify=lambda *args:Decision('safe',{'safe':.9,'unsafe':.1},{},'ok',1))
    monkeypatch.setattr('run.development.adapter_for',lambda *args:adapter)
    state=run(tmp_path,3)
    assert state['status']=='completed' and state['completed_requests']==1
    assert logged[0]['scores']['valid_output']==1
    assert logged[0]['metadata']['model']=='gpt-6-luna'
    assert logged[0]['metadata']['arm']=='decisions_luna'
    with pytest.raises(ValueError,match='already exists'):
        run(tmp_path,3)


def test_reference_scores_human_interaction_only():
    from adapters.base import Decision
    row={'expected':{'s2':'unsafe','response_s2':'unsafe'},'metadata':{'source':'aegis_v2_validation','label_source':'human','response_refusal_label':'refusal'}}
    d=Decision('safe',{'safe':.9,'unsafe':.1},{},'ok',1)
    job={'target':'interaction_harm','scheme':'s2'}
    assert scores_for('development_reference',job,d,row)['binary_correct_strict']==0
    for target in ('prompt_harm','response_harm','response_refusal'):
        job['target']=target
        assert set(scores_for('development_reference',job,d,row))=={'valid_output','refused'}


def test_bundle_uses_matched_questions_and_counts_usage_once(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import braintrust
    from adapters.base import Decision
    from run.offline_review import bundled_request
    from run.matrix import COMPANION_RUBRICS
    row,job=make_plan(tmp_path)
    job.update(target='bundle',rubric='bundle')
    job['request']=bundled_request(job['arm'],row)
    job['request_sha256']=digest(job['request'])
    (tmp_path/'manifest.jsonl').write_text(json.dumps(job)+'\n')
    events=[]
    class Span:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def log(self,**event): events.append(event)
    experiment=NS(summarize=lambda:NS(experiment_url='test'),start_span=lambda **_:Span(),flush=lambda:None)
    monkeypatch.setattr(braintrust,'init',lambda **_:experiment)
    monkeypatch.setattr('run.development.load_braintrust_key',lambda:None)
    def classify_many(text,questions):
        assert list(questions)==list(COMPANION_RUBRICS)
        assert text==row['input']['text']
        return {t:Decision(next(iter(q['criteria'])),None,{},'ok',3,{'input_tokens':10}) for t,q in questions.items()}
    monkeypatch.setattr('run.development.adapter_for',lambda *args:NS(model_version='test',classify_many=classify_many))
    assert run(tmp_path,3)['completed_requests']==1
    result=json.loads((tmp_path/'results.jsonl').read_text())
    assert len(result['batch_results'])==3
    assert result['usage']=={'input_tokens':10}
    assert result['usage_scope']=='whole_bundle_once'
    assert len(events)==1
