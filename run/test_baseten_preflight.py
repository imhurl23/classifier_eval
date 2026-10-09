"""Management safeguards must work without live credentials or paid requests."""
import json
import pytest
from run import baseten_preflight as runner


def fake_management(monkeypatch,tmp_path,mode):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'load_local_service_keys',lambda:None)
    monkeypatch.setenv('BASETEN_API_KEY','test-only')
    calls=[]
    def request(method,url,**kwargs):
        calls.append((method,url))
        deployment=url.split('/deployments/')[1].split('/')[0]
        arm=next(a for a,(_,d) in runner.ALLOWLIST.items() if d==deployment)
        model,_=runner.ALLOWLIST[arm]
        class Response:
            def raise_for_status(self):
                if mode=='activation_failure' and method=='POST' and url.endswith('/activate'):
                    raise RuntimeError('simulated activation failure')
            def json(self):
                if method=='POST':return {'success':True}
                activating=any(m=='POST' and u.endswith('/activate') for m,u in calls)
                shutting=any(m=='POST' and u.endswith('/deactivate') for m,u in calls)
                status='UNHEALTHY' if activating and not shutting else 'INACTIVE'
                gpu='H100 - 1 H100 GPU, 80 GiB VRAM' if arm=='safeguard_120b' else 'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB' if arm in {'tev1_native','qwen35_4b_letters'} else 'L4:4x16 - 1 L4 GPU, 24 GiB video memory, 4 vCPU, 16 GiB'
                if mode=='hardware_drift':gpu='B200'
                return dict(id=deployment,model_id=model,status=status,active_replica_count=0,
                            autoscaling_settings={'min_replica':0,'max_replica':1},instance_type_name=gpu)
        return Response()
    monkeypatch.setattr(runner.requests,'request',request)
    monkeypatch.setattr(runner.signal,'signal',lambda *_:None)
    return calls


def test_hardware_drift_blocks_activation(monkeypatch,tmp_path):
    calls=fake_management(monkeypatch,tmp_path,'hardware_drift')
    with pytest.raises(ValueError,match='hardware changed'):runner.main()
    assert all(method=='GET' for method,_ in calls)


@pytest.mark.parametrize('mode',['activation_failure','unhealthy'])
def test_failed_startup_deactivates_all_attempted_deployments(monkeypatch,tmp_path,mode):
    calls=fake_management(monkeypatch,tmp_path,mode)
    runner.main()
    activated={url.removesuffix('/activate') for method,url in calls if method=='POST' and url.endswith('/activate')}
    deactivated={url.removesuffix('/deactivate') for method,url in calls if method=='POST' and url.endswith('/deactivate')}
    assert activated and activated==deactivated
    receipt=next((tmp_path/'data/processed/baseten-preflight').glob('*/execution.json'))
    state=json.loads(receipt.read_text())
    assert state['status']=='stopped'
    assert all(d.get('final_replica_count',0)==0 for d in state['deployments'].values())
