from copy import deepcopy
from types import SimpleNamespace

import pytest

from run.experiment_groups import Configuration, init_experiments
from run.split_experiments import copy_event, fetch_many, row_configuration, verify_events


def test_routing_separates_interfaces_and_configurations_and_reuses_children(monkeypatch):
    import braintrust
    created, calls = [], []
    def create(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(start_span=lambda **event: calls.append((kwargs['experiment'], event)),
                               log=lambda **event: calls.append((kwargs['experiment'], event)), flush=lambda: None)
    monkeypatch.setattr(braintrust, 'init', create)
    group = init_experiments(project='decision-interface-safety', experiment='study_stage',
                             metadata={'arms':['luna_structured','decisions_luna'], 'study_status':'current'})
    checkpoint_urls = group.experiment_urls
    for arm, scheme in [('luna_structured','s2'), ('decisions_luna','s2'), ('luna_structured','s3'), ('luna_structured','s2')]:
        group.start_span(metadata={'arm':arm,'scheme':scheme,'target':'interaction_harm','rubric':'r_long'})
    assert len(created) == 3
    assert calls[0][0] == calls[3][0]
    assert calls[0][0] != calls[1][0] != calls[2][0]
    assert len(checkpoint_urls) == 3
    for experiment in created:
        metadata = experiment['metadata']
        assert len(metadata['arms']) == len(metadata['models']) == 1
        assert metadata['model'] == 'gpt-6-luna'
        assert metadata['run_group'] == 'study_stage'
        assert experiment['set_current'] is False


def test_unknown_arm_fails_before_creating_experiments(monkeypatch):
    import braintrust
    monkeypatch.setattr(braintrust,'init',lambda **_:pytest.fail('Must not create an experiment'))
    group = init_experiments(project='test', experiment='run')
    with pytest.raises(KeyError):
        group.log(metadata={'arm':'unknown'}, output={})


def test_copy_preserves_outputs_scores_latency_dates_and_trace_identity():
    source={'id':'source-id','name':'full_context_reference'}
    row={'id':'row-id','created':'2026-10-07T23:00:00Z','_xact_id':'revision',
         'input':{'conversation':'unchanged'},'output':{'label':'unsafe'},
         'expected':{'label':'unsafe'},'scores':{'exact_match':1},
         'metrics':{'start':10,'end':12,'latency_ms':2000},'metadata':{'job_id':'j'},
         'span_attributes':{'name':'decision','type':'llm','created_by_user_id':'u'},
         'span_id':'span','root_span_id':'span','span_parents':[]}
    config=Configuration('jev_choice','interaction_harm','s2','r_long')
    event=copy_event(row,source,config)
    assert event['metrics']==row['metrics'] and event['created']==row['created']
    assert event['span_id']==row['span_id'] and event['root_span_id']==row['root_span_id']
    assert event['metadata']['source_experiment_id']=='source-id'
    assert event['metadata']['copied_without_inference'] is True
    verify_events([event],[deepcopy(event)])
    normalized=deepcopy(event)
    normalized['origin']['created']='2026-10-07T23:00:00.000Z'
    verify_events([event],[normalized])
    changed=deepcopy(event);changed['scores']['exact_match']=0
    with pytest.raises(ValueError,match='scores differs'):
        verify_events([event],[changed])


def test_full_conversation_configuration_comes_from_exact_job_manifest():
    row={'id':'r','metadata':{'arm':'jev_choice','job_id':'job'},'span_attributes':{'name':'jev_choice__interaction_harm__s13'}}
    jobs={'job':{'arm':'jev_choice','target':'interaction_harm','scheme':'s13','rubric':'r_long'}}
    config=row_configuration(row,{'name':'full_context_reference'},jobs)
    assert config==Configuration('jev_choice','interaction_harm','s13','r_long')
    row['metadata']['scheme']='s2'
    with pytest.raises(ValueError,match='configuration mismatch'):
        row_configuration(row,{'name':'full_context_reference'},jobs)


def test_unknown_configuration_is_never_guessed():
    with pytest.raises(ValueError,match='Missing target'):
        row_configuration({'id':'r','metadata':{'arm':'jev_choice'}},{'name':'unknown'},{})


def test_verification_reads_more_than_one_thousand_rows(monkeypatch):
    import re
    rows=[{'id':str(i),'experiment_id':'child','_pagination_key':f'{i:05}'} for i in reversed(range(1040))]
    def query(method,path,raw_response,**kwargs):
        text=kwargs['json']['query']
        cap=int(re.search(r'LIMIT (\d+)',text).group(1))
        before=re.search(r"_pagination_key < '([^']+)'",text)
        selected=[r for r in rows if not before or r['_pagination_key'] < before.group(1)]
        return SimpleNamespace(json=lambda:{'data':selected[:cap]},headers={})
    monkeypatch.setattr('run.split_experiments.api',query)
    assert len(fetch_many(['child'])['child']) == 1040


def test_verification_waits_for_insert_visibility_and_rejects_extra_rows(monkeypatch, tmp_path):
    from run.split_experiments import wait_for_copied_rows
    (tmp_path / 'events.json').write_text('[{"id":"expected"}]')
    children = [{'id':'child', 'events':'events.json'}]
    pages = iter([{}, {'child':[{'id':'expected'}]}])
    sleeps = []
    monkeypatch.setattr('run.split_experiments.fetch_many', lambda _:next(pages))
    monkeypatch.setattr('run.split_experiments.time.sleep', sleeps.append)
    assert wait_for_copied_rows(children, tmp_path, 'study')['child'] == [{'id':'expected'}]
    assert sleeps == [2]
    monkeypatch.setattr('run.split_experiments.fetch_many', lambda _:{'child':[{'id':'unexpected'}]})
    with pytest.raises(ValueError, match='Unexpected records'):
        wait_for_copied_rows(children, tmp_path, 'study')


@pytest.mark.parametrize('corrupt', [False, True])
def test_migration_copies_before_archiving_and_never_archives_corrupt_results(monkeypatch, tmp_path, corrupt):
    import json
    import run.split_experiments as migration
    config = Configuration('jev_choice', 'interaction_harm', 's2', 'r_long')
    originals, source_rows, plan = {}, {}, []
    for index in range(2):
        source = {'id':f'source-{index}', 'name':f'study-{index}', 'metadata':{}}
        row = {'id':f'row-{index}', 'created':'2026-10-01T00:00:00Z', '_xact_id':'revision',
               'metadata':{'arm':'jev_choice'}, 'output':{'label':'safe'}, 'scores':{'valid_output':1}}
        event = copy_event(row, source, config)
        filename = f'events-{index}.json'
        (tmp_path / filename).write_text(json.dumps([event]))
        originals[source['id']] = source
        source_rows[source['id']] = [row]
        plan.append({**source, 'rows':1, 'source_sha256':migration.digest([row]),
                     'children':[{'name':config.name(source['name']), 'events':filename,
                                  'metadata':{**config.metadata(), 'study_status':'current',
                                              'source_experiment_id':source['id'], 'source_record_sha256':'record-hash'}}]})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    events, calls = {}, []
    def api(method, path, **kwargs):
        calls.append((method, path))
        if method == 'GET':
            return originals[path.split('/')[1]]
        if method == 'POST' and path == 'experiment':
            child_id = f'child-{len(events)}'
            events[child_id] = []
            return {'id':child_id, **kwargs['json']}
        if path.endswith('/insert'):
            events[path.split('/')[1]].extend(deepcopy(kwargs['json']['events']))
        return {}
    def fetch_many(ids):
        copied = {key:deepcopy(events[key]) for key in ids}
        if corrupt and copied:
            copied[next(iter(copied))][0]['scores']['valid_output'] = 0
        return copied
    monkeypatch.setattr(migration, 'api', api)
    monkeypatch.setattr(migration, 'list_experiments', lambda:[])
    monkeypatch.setattr(migration, 'fetch', source_rows.__getitem__)
    monkeypatch.setattr(migration, 'fetch_many', fetch_many)
    if corrupt:
        with pytest.raises(ValueError, match='scores differs'):
            migration.apply(tmp_path)
        assert not any(method == 'PATCH' and 'source-' in path for method, path in calls)
    else:
        migration.apply(tmp_path)
        inserts = [i for i, (_, path) in enumerate(calls) if path.endswith('/insert')]
        archives = [i for i, (method, path) in enumerate(calls) if method == 'PATCH' and 'source-' in path]
        assert len(inserts) == len(archives) == 2
        assert max(inserts) < min(archives)
        receipts = json.loads((tmp_path / 'receipt.json').read_text())
        assert len(receipts) == 2 and all(r['verified'] and r['inference_calls'] == 0 for r in receipts)
