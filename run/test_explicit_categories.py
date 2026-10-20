from copy import deepcopy
import json

import pytest

from run import explicit_categories as runner
from run import explicit_categories_batch as batch


@pytest.mark.parametrize('arm', runner.ARMS)
def test_each_arm_has_identical_inputs_and_policy_changes_only(arm):
    rows, definitions, jobs = runner.plan(arm)
    assert len(jobs) == len({j['job_id'] for j in jobs}) == 3362
    assert sum(j['source_gold'] == 'safe' for j in jobs) == 1872
    source = {r['id']: r for r in rows}
    # Check both conditions on every example without invoking an adapter.
    by_item = {}
    for job in jobs:
        by_item.setdefault(job['item_id'], {})[job['variant']] = job
    explicit_policies = set()
    for item, pair in by_item.items():
        a, b = pair['original_policy_repeat'], pair['explicit_categories']
        assert a['option_order'] == b['option_order']
        assert a['source_gold'] == b['source_gold'] == source[item]['expected']['prompt_s2']
        explicit_policies.add(b['instructions'])
        assert b['instructions'] == a['instructions'] + '\n\n' + runner.RULES.read_text().strip()
    assert len(explicit_policies) == 1
    # Rendering is deterministic for both policies, with no per-item label hints.
    for job in jobs[:2]:
        assert job['request'] == runner.make_request(arm, source[job['item_id']], job['instructions'], job['option_order'], definitions)


def test_failed_second_gpu_activation_shuts_down_every_attempted_deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'OUT', tmp_path)
    def load():
        monkeypatch.setenv('BASETEN_API_KEY', 'fake')
        for arm, prefix in [('tev1_native', 'BASETEN_TEV1'), ('qwen35_4b_letters', 'BASETEN_QWEN35')]:
            model, deployment = batch.ALLOWLIST[arm]
            monkeypatch.setenv(prefix + '_MODEL_ID', model)
            monkeypatch.setenv(prefix + '_DEPLOYMENT_ID', deployment)
    monkeypatch.setattr(batch, 'load_local_service_keys', load)
    calls = []
    class Response:
        def __init__(self, payload, fail=False): self.payload, self.fail = payload, fail
        def raise_for_status(self):
            if self.fail: raise RuntimeError('activation failed')
        def json(self): return self.payload
    def request(method, url, **kwargs):
        calls.append((method, url))
        if url.endswith('/activate'):
            return Response({}, fail=len([u for m, u in calls if u.endswith('/activate')]) == 2)
        if url.endswith('/deactivate'): return Response({})
        model, deployment = url.split('/models/')[1].split('/deployments/')
        return Response({'model_id': model, 'id': deployment, 'status': 'INACTIVE', 'active_replica_count': 0,
                         'instance_type_name': 'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB',
                         'autoscaling_settings': {'min_replica': 0, 'max_replica': 1}})
    monkeypatch.setattr(batch.requests, 'request', request)
    state = batch.run(['tev1_native', 'qwen35_4b_letters'], baseten=True)
    assert state['status'] == 'stopped'
    assert len([u for m, u in calls if u.endswith('/deactivate')]) == 2
    assert all(d['final_status'] == 'INACTIVE' and d['final_replicas'] == 0 for d in state['deployments'].values())
