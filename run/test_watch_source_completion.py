import json
import pytest
from run import watch_source_completion as watcher


def fixture_plan(tmp_path,state):
    state.setdefault('baseten_shutdown',{str(i):{'verified_inactive':True} for i in range(6)})
    (tmp_path/'summary.json').write_text(json.dumps({'version':'test','compute_reserve_usd':150}))
    (tmp_path/'baseten-execution.json').write_text(json.dumps(state))
    (tmp_path/'baseten-results.jsonl').write_text(json.dumps({'job_id':'one','arm':'tev1_native','status':'ok'})+'\n')
    (tmp_path/'manifest.jsonl').write_text(json.dumps({'job_id':'one','arm':'tev1_native'})+'\n')


def test_watcher_never_repeats_ambiguous_inflight_calls(tmp_path,monkeypatch):
    fixture_plan(tmp_path,{'status':'partial','inflight':{'unknown':{}}})
    monkeypatch.setattr(watcher,'list_experiments',lambda:[])
    monkeypatch.setattr(watcher,'run',lambda *a,**k:pytest.fail('Must not repeat an ambiguous call'))
    with pytest.raises(ValueError,match='Ambiguous'):watcher.watch(tmp_path)
    assert json.loads((tmp_path/'completion.json').read_text())['status']=='incomplete_or_unverified'


def test_failed_remote_audit_demotes_current_experiments_without_more_inference(tmp_path,monkeypatch):
    fixture_plan(tmp_path,{'status':'completed','inflight':{}})
    monkeypatch.setattr(watcher,'run',lambda *a,**k:pytest.fail('QA must not repeat model inference'))
    def bad_audit(*a,**k):raise AssertionError('Gold/request binding mismatch')
    monkeypatch.setattr(watcher,'verify',bad_audit);monkeypatch.setattr(watcher.time,'sleep',lambda _:None)
    monkeypatch.setattr(watcher,'list_experiments',lambda:[{'id':'exp','metadata':{'plan_version':'test','study_status':'current','model':'Tev1'}}])
    changes=[];monkeypatch.setattr(watcher,'api',lambda *a,**k:changes.append(k['json']))
    with pytest.raises(AssertionError):watcher.watch(tmp_path)
    assert changes[0]['metadata']['study_status']=='partial'
    assert changes[0]['metadata']['model']=='Tev1' and not changes[0]['metadata']['verification_complete']
