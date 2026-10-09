import json
from types import SimpleNamespace as NS
import pytest
from adapters.base import Decision
from run import source_benchmark as benchmark


def row(index=0):
    return {'id':f'wildguard-{index:06d}','input':{'text':'User:\nHello\n\nAssistant:\nHello',
             'prompt':'Hello','response':'Hello','unit':'conversation'},
            'metadata':{'source':'wildguardtest','label_source':'human','split':'test',
                        'response_refusal_label':'compliance','gold_provenance':'human target labels'},
            'expected':{'s2':'safe','prompt_s2':'safe','response_s2':'safe'}}


def test_plan_requires_human_labels_and_freezes_requests(tmp_path,monkeypatch):
    rows=[row(i) for i in range(10)]
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    monkeypatch.setattr(benchmark,'candidates',lambda:rows)
    monkeypatch.setattr(benchmark,'sha',lambda _: 'frozen')
    path=benchmark.prepare(25,max_items=10)
    summary, _, jobs=benchmark.load_plan(path)
    assert summary['items']==10 and summary['requests']==210
    assert summary['planning_reserve_usd'] <= 20
    assert not any(j['arm']=='decisions_luna' for j in jobs)
    changed=[json.loads(l) for l in (path/'sample.jsonl').read_text().splitlines()]
    changed[0]['metadata']['label_source']='gpt4'
    (path/'sample.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError,match='Source human label'):
        benchmark.load_plan(path)


def test_budget_failure_never_initializes_a_provider(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'candidates',lambda:[row(i) for i in range(8)])
    with pytest.raises(ValueError,match='meaningful matched cohort'):
        benchmark.prepare(.00001)
    assert not (tmp_path/'execution.json').exists()


def test_runner_checkpoints_predictions_before_logging_and_never_repeats_paid_calls(tmp_path,monkeypatch):
    import braintrust
    sample=row()
    job=next(benchmark.jobs_for(sample))
    summary={'version':'test','additional_budget_usd':25,'planning_reserve_usd':1,'items':1}
    monkeypatch.setattr(benchmark,'load_plan',lambda _:(summary,{sample['id']:sample},[job]))
    monkeypatch.setattr(benchmark,'load_braintrust_key',lambda:None)
    dataset=NS(insert=lambda **_:None,flush=lambda:None)
    monkeypatch.setattr(braintrust,'init_dataset',lambda **_:dataset)
    logs=[]
    class Span:
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def log(self,**event):
            assert (tmp_path/'results.jsonl').exists()
            assert json.loads((tmp_path/'execution.json').read_text())['inflight']=={}
            logs.append(event)
    group=NS(experiment_urls={},start_span=lambda **_:Span(),flush=lambda:None)
    monkeypatch.setattr(benchmark,'init_experiments',lambda **_:group)
    calls=[]
    def classify(*_):
        calls.append(1)
        return Decision('safe',{'safe':.9,'unsafe':.1},{},'ok',10,{'input_tokens':12,'output_tokens':2})
    monkeypatch.setattr(benchmark,'adapter_for',lambda *_:NS(model_version='test',classify=classify))
    assert benchmark.run(tmp_path)['status']=='completed'
    assert logs[0]['scores']['source_accuracy']==1
    assert logs[0]['expected']=={'label':'safe'}
    assert logs[0]['metrics']['prompt_tokens']==12
    assert logs[0]['dataset_record_id']==sample['id']
    with pytest.raises(ValueError,match='already started'):
        benchmark.run(tmp_path)
    assert calls==[1]
    state=json.loads((tmp_path/'execution.json').read_text());state['inflight']={'uncertain':{}}
    (tmp_path/'execution.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Ambiguous inflight'):
        benchmark.run(tmp_path,resume=True)
    assert calls==[1]
