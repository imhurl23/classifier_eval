"""Restore verified source-labeled development results without repeating inference."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import time
import json
from pathlib import Path

from data.bt_auth import ROOT, load_braintrust_key
from run.configure_braintrust_audit import PROJECT
from run.source_labels import BENCHMARK, CONTEXT_POLICY, source_gold, source_scores, usage_metrics
from run.split_experiments import api, fetch_many, list_experiments

REFERENCE = ROOT / 'data/processed/study-reference/00de7797f0a6'
RUN_GROUP = 'development_reference__00de7797f0a6__20261007T165032Z'


def restore():
    import braintrust as bt
    load_braintrust_key()
    experiments = list_experiments()
    primary = [e for e in experiments if (e.get('metadata') or {}).get('run_group') == RUN_GROUP
               and e['metadata'].get('target') == 'interaction_harm'
               and e['metadata'].get('experiment_layout') == 'single_arm_configuration_v1']
    if len(primary) != 7:
        raise ValueError('Expected seven completed source-labeled model/interface experiments')
    rows = {r['id']: r for r in map(json.loads, (REFERENCE / 'sample.jsonl').read_text().splitlines())}
    results = {r['job_id']: r for r in map(json.loads, (REFERENCE / 'results.jsonl').read_text().splitlines())}
    dataset_info = json.loads((ROOT / 'run_manifest.json').read_text())['datasets']['dis-dev']
    datasets = api('GET', 'dataset', params={'project_id':PROJECT, 'limit':100})['objects']
    dataset = next(d for d in datasets if d['name'] == dataset_info['braintrust_name'])
    dataset_binding = {'id':dataset['id'], 'version':dataset_info['braintrust_version']}
    fetched = fetch_many([e['id'] for e in primary])
    directory = ROOT / 'data/processed/source-label-restoration' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    (directory / 'experiments-before.json').write_text(json.dumps(experiments, indent=2)+'\n')
    before = deepcopy(fetched)
    (directory / 'rows-before.json').write_text(json.dumps(before)+'\n')
    patches = {}
    for experiment in primary:
        events = fetched.get(experiment['id'], [])
        if len(events) != 200:
            raise ValueError('Reference coverage changed')
        changes = []
        for event in events:
            result = results[event['metadata']['job_id']]
            row = rows[result['item_id']]
            if result['arm'] != experiment['metadata']['arm'] or event['metadata']['request_sha256'] != result['request_sha256']:
                raise ValueError('Existing prediction does not match pinned request')
            output = {k: result.get(k) for k in ('label', 'probs', 'status')}
            if event.get('output') != output or source_gold(row, result['target']) is None:
                raise ValueError('Source labels or prediction binding changed')
            scores = source_scores(output, row, result['target'], result['scheme'])
            if any(event.get('scores', {}).get(k) != v for k, v in scores.items() if k != 'source_accuracy'):
                raise ValueError('Existing source scores disagree with recomputation')
            changes.append({'id': event['id'], 'scores': scores,
                'metrics': usage_metrics(result.get('usage')), 'dataset_record_id':row['id'],
                'metadata': {'benchmark_id': BENCHMARK, 'source_dataset': row['metadata']['source'],
                             'source_item_id': row['id'], 'source_label_provenance': row['metadata']['gold_provenance'],
                             'source_label_scope': row['metadata']['source_label_scope'],
                             'human_gold_available': True, 'scoring_basis': 'published_source_labels',
                             'predictions_reused': True}})
        patches[experiment['id']] = changes
    # Finish checking every binding before performing any write.
    (directory / 'span-patches.json').write_text(json.dumps(patches)+'\n')
    for experiment in primary:
        writer = bt.init(project_id=PROJECT, experiment=experiment['name'], update=True, set_current=False, dataset=dataset_binding)
        for event in patches[experiment['id']]:
            writer.update_span(**event)
        writer.flush()
    time.sleep(5)  # Event updates may become query-visible after the insert receipt.
    after = fetch_many([e['id'] for e in primary])
    for experiment in primary:
        original = {r['id']: r for r in before[experiment['id']]}
        checked = {r['id']: r for r in after[experiment['id']]}
        if checked.keys() != original.keys():
            raise ValueError('Restoration changed row identities')
        for patch in patches[experiment['id']]:
            event = checked[patch['id']]
            for key in ('input','output','created','span_id','root_span_id','span_parents','origin','tags'):
                old, new = deepcopy(original[patch['id']].get(key)), deepcopy(event.get(key))
                if key == 'origin':
                    for value in (old, new):
                        if value and value.get('created'):
                            value['created'] = datetime.fromisoformat(value['created'])
                if old != new:
                    raise ValueError('Restoration changed original '+key)
            if any(event.get('scores', {}).get(k) != v for k,v in patch['scores'].items()):
                raise ValueError('Restored scores are not visible yet; rerun safely')
            if any(event.get('metadata', {}).get(k) != v for k,v in patch['metadata'].items()):
                raise ValueError('Restored label provenance is not visible yet; rerun safely')
            if any(event.get('metrics', {}).get(k) != v for k,v in patch['metrics'].items()):
                raise ValueError('Recorded usage was not restored')
    current_ids = {e['id'] for e in primary}
    changes = []
    for experiment in experiments:
        metadata = experiment.get('metadata') or {}
        if experiment['id'] in current_ids:
            new = {**metadata, 'study_status':'current', 'benchmark_id':BENCHMARK,
                   'benchmark_primary':True, 'human_gold_available':True,
                   'evaluation_role':'source_labeled_development', 'evaluation_split':'validation',
                   'source_dataset':'aegis_v2_validation', 'scoring_basis':'published_source_labels',
                   'context_policy':CONTEXT_POLICY, 'superseded_for_current_study':False,
                   'predictions_reused':True, 'manual_annotation_required':False}
        elif metadata.get('experiment_layout') == 'single_arm_configuration_v1' and (
                experiment['name'].startswith('full_context_') or metadata.get('run_group') == RUN_GROUP or
                experiment['name'].startswith(('development_repeatability__','development_batching__'))):
            new = {**metadata, 'study_status':'supplemental', 'benchmark_primary':False,
                   'evaluation_role':'full_context_diagnostic' if experiment['name'].startswith('full_context_') else 'diagnostic',
                   'manual_annotation_required':False}
        else:
            continue
        api('PATCH','experiment/'+experiment['id'],json={'metadata':new, **({'description':'Source-labeled Aegis validation benchmark. Published human interaction labels; original predictions reused and scores recomputed. Development source agreement, not held-out performance.'} if experiment['id'] in current_ids else {})})
        changes.append({'id':experiment['id'], 'name':experiment['name'], 'study_status':new['study_status']})
    report = {'directory':str(directory), 'benchmark_id':BENCHMARK, 'primary_experiments':len(primary),
              'source_labeled_examples':len(rows), 'predictions_reused':sum(len(p) for p in patches.values()),
              'inference_calls':0, 'scoring_verified':True, 'metadata_changes':changes,
              'scope':'Aegis validation source-label agreement; development results, not held-out performance.'}
    (directory / 'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metadata_changes'}), flush=True)
    return report


if __name__ == '__main__':
    restore()
