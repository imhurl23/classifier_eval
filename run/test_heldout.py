"""No paid calls: audit completeness, budget arithmetic, derivations and restart safety."""
import copy,json
from pathlib import Path
import pytest
from adapters.base import Decision
from run import heldout
from run.freeze_heldout import validate_audit,validate_budget


def complete_audit():
    packet=json.loads((heldout.ROOT/'data/processed/offline-review/f0c7bbda4b65/annotation-template.json').read_text())
    packet['rater_id']='test-reviewer'
    for r in packet['ratings']:
        r.update(interaction_harm='safe',prompt_harm='safe',response_harm='safe',response_refusal='not_refusal')
    return packet


def test_single_reviewer_audit_requires_complete_unique_ratings_and_reasons():
    audit=complete_audit();assert validate_audit(audit) is audit
    bad=copy.deepcopy(audit);bad['ratings'][-1]=bad['ratings'][0]
    with pytest.raises(ValueError,match='unique'):validate_audit(bad)
    bad=copy.deepcopy(audit);bad['ratings'][0]['response_harm']='unrateable'
    with pytest.raises(ValueError,match='reason'):validate_audit(bad)
    bad['ratings'][0]['reason']='Insufficient context';validate_audit(bad)


def test_budget_accounts_for_past_spend_compute_and_fees():
    summary={'requests_by_arm':{'guard':1},'artifact_sha256':{'manifest.jsonl':'abc'}}
    b={'manifest_sha256':'abc','billing_verified':True,'total_study_planning_ceiling_usd':1000,'max_wall_minutes':30,
       'prior_spend_and_unsettled_ceiling_usd':800,'prior_spend_evidence':'ledger', 'shared_fees_and_contingency_usd':25,
       'arms':{'guard':{'rate_basis':'GPU minutes','evidence':'provider price','ceiling_usd':175}}}
    assert validate_budget(b,summary)==(1000,30)
    b['arms']['guard']['ceiling_usd']=176
    with pytest.raises(ValueError,match='exceed'):validate_budget(b,summary)
    b['arms']['guard']['ceiling_usd']=float('nan')
    with pytest.raises(ValueError,match='ceilings'):validate_budget(b,summary)


def example_row():
    return json.loads((heldout.ROOT/'data/processed/heldout-candidate/55a1f8b24835/sample.jsonl').read_text().splitlines()[0])


def part(arm,row,target,label,status='ok'):
    return {'job_id':target,'arm':arm,'item_id':row['id'],'target':target,'label':label,'status':status,'latency_ms':10}


def test_native_or_requires_both_valid_turns_and_preserves_caution():
    row=example_row();rows={row['id']:row}
    a=part('qwen3guard_8b',row,'prompt_harm','safe');b=part('qwen3guard_8b',row,'response_harm','needs_caution')
    d=heldout.derive_native([a,b],rows)[0]
    assert d['label']=='needs_caution' and d['physical_calls']==0 and d['scheme']=='s3'
    assert heldout.derive_native([a],rows)[0]['status']=='invalid_output'
    b.update(label=None,status='api_error')
    assert heldout.derive_native([a,b],rows)[0]['label'] is None
    b.update(label='unsafe',status='ok')
    assert heldout.derive_native([a,b],rows)[0]['label']=='unsafe'


def test_shared_native_call_latency_and_usage_not_duplicated():
    row=example_row();p={'job_id':'pair','arm':'nemotron_guard','item_id':row['id'],'target':'native_pair','latency_ms':40,
      'batch_results':{t:{'label':'safe','status':'ok','latency_ms':40} for t in ('prompt_harm','response_harm')}}
    d=heldout.derive_native([p],{row['id']:row})[0]
    assert d['latency_ms']==40 and d['usage']=={} and d['parent_job_ids']==['pair']


def test_unfrozen_plan_rejected_before_credential_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(heldout,'load_plan',lambda _:({}, {}, []))
    monkeypatch.setattr(heldout,'load_local_service_keys',lambda:pytest.fail('Credentials loaded before gate'))
    with pytest.raises(ValueError,match='Not frozen'):heldout.run(tmp_path)


def test_ambiguous_paid_call_cannot_resume(monkeypatch,tmp_path):
    monkeypatch.setattr(heldout,'load_plan',lambda _:({}, {}, [{'job_id':'paid'}]))
    monkeypatch.setattr(heldout,'verify_launch',lambda *_:{'max_wall_minutes':30})
    (tmp_path/'execution.json').write_text(json.dumps({'status':'partial','inflight':{'paid':{'arm':'test'}}}))
    monkeypatch.setattr(heldout,'load_local_service_keys',lambda:pytest.fail('Paid call repeated'))
    with pytest.raises(ValueError,match='Ambiguous paid calls'):heldout.run(tmp_path,30,resume=True)


def test_checkpoint_survives_logging_failure_and_resume_skips_paid_result(monkeypatch,tmp_path):
    import sys,types
    row=example_row();jobs=[dict(job_id=str(n),arm='jev_choice',item_id=row['id'],target='interaction_harm',scheme='s2',rubric='r_long',replicate=0,request={},request_sha256='request') for n in range(2)]
    monkeypatch.setattr(heldout,'load_plan',lambda _:({'version':'test'},{row['id']:row},jobs))
    monkeypatch.setattr(heldout,'verify_launch',lambda *_:{'max_wall_minutes':30})
    monkeypatch.setattr(heldout,'load_local_service_keys',lambda:None)
    monkeypatch.setattr(heldout,'load_braintrust_key',lambda:None)
    (tmp_path/'freeze.json').write_text('{}')
    called=[];fail_logging=[True]
    class Span:
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def log(self,**_):
            if fail_logging[0]:fail_logging[0]=False;raise RuntimeError('Log unavailable')
    class Experiment:
        def summarize(self):return types.SimpleNamespace(experiment_url='test')
        def start_span(self,**_):return Span()
        def flush(self):pass
    monkeypatch.setitem(sys.modules,'braintrust',types.SimpleNamespace(init=lambda **_:Experiment()))
    def execute(job,*_):
        called.append(job['job_id']);return Decision('safe',None,{},'ok',10),'test',None
    monkeypatch.setattr(heldout,'execute',execute)
    monkeypatch.setattr(heldout,'result_scores',lambda *_:{})
    state=heldout.run(tmp_path,30)
    assert state['status']=='partial' and state['completed_requests']==1 and state['inflight']=={}
    state=heldout.run(tmp_path,30,resume=True)
    assert state['status']=='completed' and called==['0','1']
    assert len((tmp_path/'results.jsonl').read_text().splitlines())==2


def test_deployment_drift_blocks_before_inference(monkeypatch):
    model,deployment=heldout.ALLOWLIST['tev1_native']
    monkeypatch.setenv('BASETEN_TEV1_MODEL_ID',model)
    monkeypatch.setenv('BASETEN_TEV1_DEPLOYMENT_ID','changed')
    with pytest.raises(ValueError,match='deployment mismatch'):heldout.verify_baseten_binding({'tev1_native'})
    monkeypatch.setenv('BASETEN_TEV1_DEPLOYMENT_ID',deployment)
    heldout.verify_baseten_binding({'tev1_native'})
