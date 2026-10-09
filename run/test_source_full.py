import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from run import source_full as full
from run.source_labels import source_scores
from adapters.base import Decision


def sample(source='wildguardtest',unit='conversation',index=0):
    return {'id':f'{source}-{index}','input':{'text':'User:\nA\u2028B\n\nAssistant:\nReply','prompt':'A\u2028B','response':'Reply','unit':unit},
            'expected':{'s2':'unsafe','prompt_s2':'unsafe','response_s2':'safe'},
            'metadata':{'source':source,'label_source':'human','response_refusal_label':'refusal','gold_provenance':'human'}}


def test_jsonl_unicode_separator_is_content_not_a_new_record(tmp_path):
    path=tmp_path/'sample.jsonl';row=sample()
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    assert full.read_jsonl(path)==[row]


def test_full_plan_covers_all_interfaces_and_only_supported_gold():
    aegis=sample('aegis_v2_test');wild=sample();control=sample('xstest','prompt_only')
    assert len(list(full.full_jobs(aegis)))==7
    assert len(list(full.full_jobs(wild)))==25
    assert len(list(full.full_jobs(control)))==6
    wild['metadata']['response_refusal_label']=None
    jobs=list(full.full_jobs(wild))
    assert len(jobs)==19 and {j['target'] for j in jobs}=={'interaction_harm','prompt_harm','response_harm'}
    assert full.cohort_key(aegis)!=full.cohort_key(sample('aegis_v2_test','prompt_only'))


@pytest.mark.parametrize('target,gold', [('interaction_harm','unsafe'),('prompt_harm','unsafe'),('response_harm','safe'),('response_refusal','refusal')])
@pytest.mark.parametrize('status', ['ok','api_error','invalid_output','truncated','refusal'])
def test_strict_scoring_is_source_gold_agreement_even_for_failures(target,gold,status):
    result=source_scores({'label':gold,'status':status},sample(),target)
    assert result['source_accuracy']==float(status=='ok')
    assert source_scores({'label':'not_refusal' if target=='response_refusal' else 'safe' if gold=='unsafe' else 'unsafe','status':'ok'},sample(),target)['source_accuracy']==0


def test_native_ternary_turns_preserve_abstention_and_binary_gold():
    for target in ('prompt_harm','response_harm'):
        scores=source_scores({'label':'needs_caution','status':'ok'},sample(),target,'s3')
        assert scores['source_accuracy']==0 and scores['needs_caution_rate']==1


def test_checkpoint_precedes_log_and_completed_job_is_never_paid_twice(tmp_path,monkeypatch):
    import braintrust
    row=sample();job=next(full.full_jobs(row))
    summary={'additional_budget_usd':1000,'request_reserve_usd':1,'version':'test'}
    monkeypatch.setattr(full,'load_plan',lambda _:(summary,{row['id']:row},[job]))
    monkeypatch.setattr(full,'load_braintrust_key',lambda:None)
    monkeypatch.setattr(braintrust,'init_dataset',lambda **_:NS(insert=lambda **_:None,flush=lambda:None))
    logs=[]
    def log(**event):
        assert full.read_jsonl(tmp_path/'results.jsonl')[0]['job_id']==job['job_id']
        assert not json.loads((tmp_path/'execution.json').read_text())['inflight']
        logs.append(event)
    group=NS(log=log,flush=lambda:None,experiment_urls={})
    monkeypatch.setattr(full,'init_experiments',lambda **_:group)
    monkeypatch.setattr(full,'replay',lambda *_:None)
    calls=[]
    def classify(*_):
        calls.append(1)
        return Decision('unsafe',{'unsafe':.9,'safe':.1},{},'ok',10,{'input_tokens':2})
    monkeypatch.setattr(full,'adapter_for',lambda *_:NS(classify=classify,model_version='test'))
    assert full.run(tmp_path,workers_per_arm=1)['status']=='completed'
    assert logs[0]['scores']['source_accuracy']==1
    with pytest.raises(ValueError,match='already started'):full.run(tmp_path)
    assert calls==[1]
