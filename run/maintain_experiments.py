"""Backfill model metadata and label legacy runs without deleting results.

Prepare a local backup first, then apply only reviewed metadata/description changes.
No inference calls are made. Existing experiment names and span IDs are preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import requests
import time

from data.bt_auth import load_braintrust_key, ROOT
from run.configure_braintrust_audit import PROJECT
from run.model_metadata import ARM_MODELS, experiment_model_metadata, model_metadata


def api(method, path, raw_response=False, **kwargs):
    url = 'https://api.braintrust.dev/btql' if path == 'btql' else 'https://api.braintrust.dev/v1/' + path
    for attempt in range(6):
        response = requests.request(method, url,
            headers={'Authorization': 'Bearer ' + load_braintrust_key()}, timeout=30, **kwargs)
        if response.status_code not in (429, 502, 503, 504) or attempt == 5:
            response.raise_for_status()
            return response if raw_response else response.json()
        delay = min(45, max(2, int(response.headers.get('Retry-After', 2 ** (attempt + 1)))))
        print(json.dumps({'api_retry_status': response.status_code, 'wait_seconds': delay}), flush=True)
        time.sleep(delay)


def span_arm(row, experiment):
    metadata = row.get('metadata') or {}
    arm = metadata.get('arm')
    if arm in ARM_MODELS:
        return arm
    name = (row.get('span_attributes') or {}).get('name', '')
    for candidate in sorted(ARM_MODELS, key=len, reverse=True):
        if name == candidate or name.startswith(candidate + '__') or row['id'].startswith(candidate + '__'):
            return candidate
    em = experiment.get('metadata') or {}
    arms = em.get('arms') or ([em['arm']] if em.get('arm') else [])
    if len(arms) == 1:
        return arms[0]
    if experiment['name'].startswith('policylm_prompt_pilot__'):
        return 'policylm_17b_prompt'
    raise ValueError('Cannot resolve model for span ' + row['id'])


def prepare():
    import braintrust as bt
    load_braintrust_key()
    experiments = []
    starting_after = None
    while True:
        params = {'project_id': PROJECT, 'limit': 100}
        if starting_after:
            params['starting_after'] = starting_after
        page = api('GET', 'experiment', params=params)['objects']
        experiments.extend(e for e in page if (e.get('metadata') or {}).get('experiment_layout') != 'original_aggregate')
        if len(page) < 100:
            break
        starting_after = page[-1]['id']
    directory = ROOT / 'data/processed/experiment-maintenance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    (directory / 'experiments-before.json').write_text(json.dumps(experiments, indent=2) + '\n')

    def inspect(experiment):
        reader = bt.init(project_id=PROJECT, experiment=experiment['name'], open=True, set_current=False)
        rows = list(reader.fetch())
        (directory / (experiment['id'] + '-before.json')).write_text(json.dumps(rows) + '\n')
        arms = sorted({span_arm(row, experiment) for row in rows})
        if not arms:
            arms = (experiment.get('metadata') or {}).get('arms', [])
        metadata = {**(experiment.get('metadata') or {}), **experiment_model_metadata(arms)}
        current = experiment['name'].startswith('full_context_')
        status = 'current' if current else 'legacy'
        metadata.update(study_status=status, superseded_for_current_study=not current)
        prefix = 'CURRENT FULL-CONVERSATION STUDY' if current else 'LEGACY PAIR-ONLY / DIAGNOSTIC — superseded for the current full-conversation study'
        description = prefix + '. Original results preserved. ' + (experiment.get('description') or '')
        patches = [{'id': row['id'], 'metadata': model_metadata(span_arm(row, experiment))}
                   for row in rows if any((row.get('metadata') or {}).get(k) != v
                       for k, v in model_metadata(span_arm(row, experiment)).items())]
        return {'id': experiment['id'], 'name': experiment['name'], 'metadata': metadata,
                'description': description.strip(), 'span_patches': patches, 'rows': len(rows)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        changes = list(pool.map(inspect, experiments))
    (directory / 'plan.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(json.dumps({'directory': str(directory), 'experiments': len(changes),
        'spans_to_update': sum(len(c['span_patches']) for c in changes),
        'current': sum(c['metadata']['study_status'] == 'current' for c in changes)}), flush=True)
    return directory


def apply(directory):
    import braintrust as bt
    load_braintrust_key()
    directory = Path(directory)
    changes = json.loads((directory / 'plan.json').read_text())
    results = []
    for change in changes:
        live = api('GET', 'experiment/' + change['id'])
        if live['project_id'] != PROJECT or live['name'] != change['name']:
            raise ValueError('Experiment binding changed')
        writer = bt.init(project_id=PROJECT, experiment=change['name'], update=True, set_current=False)
        for patch in change['span_patches']:
            writer.update_span(**patch)
        writer.flush()
        api('PATCH', 'experiment/' + change['id'], json={
            'metadata': {**(live.get('metadata') or {}), **change['metadata']},
            'description': change['description']})
        # Verify result data and row identity; only metadata may differ.
        reader = bt.init(project_id=PROJECT, experiment=change['name'], open=True, set_current=False)
        after = {r['id']: r for r in reader.fetch()}
        before = {r['id']: r for r in json.loads((directory / (change['id'] + '-before.json')).read_text())}
        if after.keys() != before.keys():
            raise ValueError('Span count or identities changed')
        for rid, old in before.items():
            new = after[rid]
            for field in ('input', 'output', 'expected', 'scores', 'metrics', 'tags', 'span_id', 'root_span_id'):
                if old.get(field) != new.get(field):
                    raise ValueError('Result changed: ' + rid + ' ' + field)
            for key, value in (old.get('metadata') or {}).items():
                if new['metadata'].get(key) != value:
                    raise ValueError('Existing metadata changed: ' + rid + ' ' + key)
            expected = model_metadata(span_arm(new, change))
            if any(new['metadata'].get(k) != v for k, v in expected.items()):
                raise ValueError('Model metadata did not persist')
        checked = api('GET', 'experiment/' + change['id'])
        if checked['metadata']['model'] != change['metadata']['model']:
            raise ValueError('Experiment model did not persist')
        results.append({'id': change['id'], 'name': change['name'], 'rows': len(after),
                        'updated_spans': len(change['span_patches']), 'verified': True})
        (directory / 'receipt.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', type=Path)
    args = parser.parse_args()
    apply(args.apply) if args.apply else prepare()
