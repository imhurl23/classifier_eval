"""Copy existing decisions into single-arm/configuration experiments, with QA.

Source experiments are preserved and marked as original aggregate archives only
after every child is verified. Copies retain outputs, scores, metrics, dates and
trace IDs, and link back to their source. No model inference is performed.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

from data.bt_auth import ROOT
from data.jsonl import read_jsonl
from run.configure_braintrust_audit import PROJECT
from run.experiment_groups import configuration
from run.maintain_experiments import api, span_arm

LAYOUT = 'single_arm_configuration_v1'
COPY_FIELDS = ('id', 'created', 'input', 'output', 'expected', 'error', 'scores',
               'metadata', 'tags', 'metrics', 'context', 'span_attributes', 'origin',
               'facets', 'span_id', 'root_span_id', 'span_parents')
VERIFY_FIELDS = tuple(k for k in COPY_FIELDS if k not in ('metadata', 'origin'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fetch(experiment_id):
    rows = []
    cursor = None
    while True:
        payload = {'limit': 1000}
        if cursor:
            payload['cursor'] = cursor
        page = api('POST', 'experiment/' + experiment_id + '/fetch', json=payload)
        rows.extend(page['events'])
        cursor = page.get('cursor')
        if not cursor:
            return rows


def fetch_many(experiment_ids):
    """Use one SQL source for several experiments to minimize rate-limited reads."""
    if not experiment_ids:
        return {}
    source = ','.join("'" + experiment_id + "'" for experiment_id in experiment_ids)
    # This organization's SQL row cap is 1,000. Keyset pagination also handles
    # a full 1,000-row response without a byte-overflow cursor.
    query = 'SELECT * FROM experiment(' + source + ')'
    rows = defaultdict(list)
    before_key = None
    seen = set()
    while True:
        page_query = query + (" WHERE _pagination_key < '" + before_key.replace("'", "''") + "'" if before_key else '')
        page_query += ' ORDER BY _pagination_key DESC LIMIT 1000'
        response = api('POST', 'btql', raw_response=True, json={'query': page_query, 'fmt': 'json'})
        data = response.json()['data']
        for row in data:
            identity = (row['experiment_id'], row['id'])
            if identity in seen:
                raise ValueError('SQL pagination repeated a record')
            seen.add(identity)
            rows[row['experiment_id']].append(row)
        cursor = response.headers.get('x-bt-cursor') or response.headers.get('x-amz-meta-bt_cursor')
        if not data or (len(data) < 1000 and not cursor):
            return rows
        before_key = data[-1]['_pagination_key']


def list_experiments():
    rows = []
    starting_after = None
    while True:
        params = {'project_id': PROJECT, 'limit': 100}
        if starting_after:
            params['starting_after'] = starting_after
        page = api('GET', 'experiment', params=params)['objects']
        rows.extend(page)
        if len(page) < 100:
            return rows
        starting_after = page[-1]['id']


def manifest_configurations():
    jobs = {}
    for path in (ROOT / 'data/processed').glob('**/manifest.jsonl'):
        for job in read_jsonl(path):
            if job.get('job_id') and all(job.get(k) for k in ('arm', 'target', 'scheme', 'rubric')):
                config = {k: job[k] for k in ('arm', 'target', 'scheme', 'rubric')}
                if job['job_id'] in jobs and jobs[job['job_id']] != config:
                    raise ValueError('Conflicting job configurations')
                jobs[job['job_id']] = config
    return jobs


def row_configuration(row, source, jobs):
    metadata = dict(row.get('metadata') or {})
    job = jobs.get(metadata.get('job_id'), {})
    arm = span_arm(row, source)
    if job and job['arm'] != arm:
        raise ValueError('Manifest/record arm mismatch')
    for key in ('target', 'scheme', 'rubric'):
        if metadata.get(key) and job.get(key) and metadata[key] != job[key]:
            raise ValueError('Manifest/record configuration mismatch')
    config = {**job, **{k: v for k, v in metadata.items() if k in ('target', 'scheme', 'rubric')}, 'arm': arm}
    em = source.get('metadata') or {}
    if source['name'].startswith('policylm_prompt_pilot__'):
        config.update(target='prompt_harm', scheme='s2', rubric='custom_policy')
    if metadata.get('question_bundle') and not config.get('rubric'):
        config['rubric'] = 'batched_turn'
    for key in ('target', 'scheme', 'rubric'):
        if not config.get(key) and em.get(key):
            config[key] = em[key]
    if not config.get('target'):
        if em.get('purpose') == 'adapter_smoke':
            config['target'] = 'interaction_harm'
        else:
            raise ValueError('Missing target for ' + row['id'])
    if not config.get('rubric'):
        raise ValueError('Missing rubric for ' + row['id'])
    return configuration(config)


def copy_event(row, source, config):
    event = {k: deepcopy(row[k]) for k in COPY_FIELDS if row.get(k) is not None}
    # Server-managed attribution belongs to the original record, not the copy.
    attrs = event.get('span_attributes', {})
    for key in ('created_by_api_key_id', 'created_by_user_id', 'exec_counter'):
        attrs.pop(key, None)
    event['metadata'] = {**(row.get('metadata') or {}),
        **{k: v for k, v in config.metadata().items() if k in ('arm', 'model', 'target', 'scheme', 'rubric')},
        'run_group': source['name'], 'source_experiment_id': source['id'],
        'source_experiment_name': source['name'], 'source_row_id': row['id'],
        'source_xact_id': row.get('_xact_id'), 'copied_without_inference': True}
    if row.get('origin'):
        event['metadata']['original_origin'] = row['origin']
    event['origin'] = {'object_id': source['id'],
                       'id': row['id'], '_xact_id': row['_xact_id'], 'created': row['created']}
    return event


def verify_events(expected, actual):
    expected_by_id = {r['id']: r for r in expected}
    actual_by_id = {r['id']: r for r in actual}
    if len(expected_by_id) != len(expected) or len(actual_by_id) != len(actual):
        raise ValueError('Duplicate row IDs')
    if expected_by_id.keys() != actual_by_id.keys():
        raise ValueError('Copied rows differ')
    for rid, event in expected_by_id.items():
        copied = actual_by_id[rid]
        for key in VERIFY_FIELDS:
            old, new = event.get(key), copied.get(key)
            if key == 'span_attributes':
                old = {k: v for k, v in (old or {}).items()
                       if k not in ('created_by_api_key_id', 'created_by_user_id', 'exec_counter')}
                new = {k: v for k, v in (new or {}).items()
                       if k not in ('created_by_api_key_id', 'created_by_user_id', 'exec_counter')}
            if key == 'created' and old and new:
                old, new = datetime.fromisoformat(old), datetime.fromisoformat(new)
            if old != new:
                raise ValueError('Copied ' + key + ' differs for ' + rid)
        for key, value in event['metadata'].items():
            if copied['metadata'].get(key) != value:
                raise ValueError('Copied metadata differs for ' + rid + ': ' + key)
        old_origin, new_origin = deepcopy(event.get('origin')), deepcopy(copied.get('origin'))
        for origin in (old_origin, new_origin):
            if origin and origin.get('created'):
                origin['created'] = datetime.fromisoformat(origin['created'])
        if new_origin != old_origin:
            raise ValueError('Copied provenance differs')


def prepare():
    sources = [e for e in list_experiments() if (e.get('metadata') or {}).get('experiment_layout') not in (LAYOUT, 'original_aggregate')]
    directory = ROOT / 'data/processed/experiment-split' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    (directory / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    jobs = manifest_configurations()

    def inspect(source):
        rows = fetch(source['id'])
        (directory / (source['id'] + '-before.json')).write_text(json.dumps(rows) + '\n')
        groups = defaultdict(list)
        for row in rows:
            if row.get('comments') or row.get('classifications'):
                raise ValueError('Manual annotations require explicit migration: ' + source['name'])
            groups[row_configuration(row, source, jobs)].append(row)
        children = []
        for config, records in sorted(groups.items(), key=lambda item: item[0].name('')):
            span_ids = {r['span_id'] for r in records}
            if any(parent not in span_ids for r in records for parent in (r.get('span_parents') or [])):
                raise ValueError('Cannot split a trace across configurations')
            name = config.name(source['name'])
            metadata = {**(source.get('metadata') or {}), **config.metadata(),
                'run_group': source['name'], 'source_experiment_id': source['id'],
                'source_experiment_name': source['name'], 'copied_without_inference': True,
                'source_record_count': len(records), 'source_record_sha256': digest(records),
                'migration_verified': False}
            event_file = digest(name)[:20] + '.json'
            events = [copy_event(r, source, config) for r in records]
            (directory / event_file).write_text(json.dumps(events) + '\n')
            children.append({'name': name, 'metadata': metadata, 'events': event_file, 'rows': len(events)})
        print(json.dumps({'prepared_source': source['name'], 'rows': len(rows), 'children': len(children)}), flush=True)
        return {'id': source['id'], 'name': source['name'], 'rows': len(rows),
                'source_sha256': digest(rows), 'children': children}

    with ThreadPoolExecutor(max_workers=4) as pool:
        plan = list(pool.map(inspect, sources))
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'directory': str(directory), 'sources': len(plan),
        'children': sum(len(s['children']) for s in plan), 'rows': sum(s['rows'] for s in plan)}), flush=True)
    return directory


def wait_for_copied_rows(children, directory, source_name):
    """SQL reads can lag inserts; only absent IDs warrant a visibility retry."""
    expected = {c['id']: {e['id'] for e in json.loads((directory / c['events']).read_text())}
                for c in children}
    for attempt in range(7):
        rows = fetch_many(list(expected))
        visible = True
        for experiment_id, expected_ids in expected.items():
            actual_ids = {e['id'] for e in rows.get(experiment_id, [])}
            if not actual_ids <= expected_ids:
                raise ValueError('Unexpected records in destination')
            visible = visible and actual_ids == expected_ids
        if visible:
            return rows
        if attempt < 6:
            delay = min(30, 2 ** (attempt + 1))
            print(json.dumps({'waiting_for_insert_visibility': source_name, 'seconds': delay}), flush=True)
            time.sleep(delay)
    raise ValueError('Copied rows remain missing after insert visibility retries')


def apply(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    existing = {e['name']: e for e in list_experiments()}
    receipt_path = directory / 'receipt.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    completed = {r['source_id']: r for r in receipts}
    pending_sources = []
    for source in plan:
        if source['id'] in completed:
            continue
        # A new score, row or annotation after preparation must never be discarded.
        live_rows = fetch(source['id'])
        if digest(live_rows) != source['source_sha256']:
            raise ValueError('Source changed after preparation: ' + source['name'])
        original = api('GET', 'experiment/' + source['id'])
        existing_rows = fetch_many([existing[c['name']]['id'] for c in source['children'] if c['name'] in existing])
        children = []
        for child in source['children']:
            events = json.loads((directory / child['events']).read_text())
            binding = existing.get(child['name'])
            if binding:
                em = binding.get('metadata') or {}
                if em.get('source_experiment_id') != source['id'] or em.get('source_record_sha256') != child['metadata']['source_record_sha256']:
                    raise ValueError('Destination name already belongs to different results')
            else:
                payload = {'project_id': PROJECT, 'name': child['name'], 'metadata': {**child['metadata'], 'study_status': 'migration_pending'},
                    'description': 'One model/interface and evaluation configuration. Results copied from ' + source['name'] + '; no new inference.',
                    'ensure_new': False}
                for key in ('dataset_id', 'dataset_version', 'repo_info', 'tags', 'public'):
                    if original.get(key) is not None:
                        payload[key] = original[key]
                binding = api('POST', 'experiment', json=payload)
                existing[child['name']] = binding
            destination_id = binding['id']
            actual = existing_rows.get(destination_id, [])
            if actual:
                known = {r['id']: r for r in actual}
                if not known.keys() <= {r['id'] for r in events}:
                    raise ValueError('Unexpected records in destination')
                verify_events([e for e in events if e['id'] in known], actual)
                pending = [e for e in events if e['id'] not in known]
            else:
                pending = events
            for offset in range(0, len(pending), 100):
                api('POST', 'experiment/' + destination_id + '/insert', json={'events': pending[offset:offset + 100]})
            children.append({'id': destination_id, 'name': child['name'], 'rows': len(events), 'events': child['events']})
        pending_sources.append((source, original, children))
        print(json.dumps({'copied_source': source['name'], 'children': len(children), 'rows': source['rows']}), flush=True)

    # Let the index catch up while other sources are copied, instead of waiting
    # after every small experiment. Originals stay visible until verified.
    for source, original, children in pending_sources:
        copied_rows = wait_for_copied_rows(children, directory, source['name'])
        for child, checked in zip(source['children'], children):
            verify_events(json.loads((directory / child['events']).read_text()), copied_rows.get(checked['id'], []))
            api('PATCH', 'experiment/' + checked['id'], json={'metadata': {**child['metadata'], 'migration_verified': True}})
            checked['verified'] = True
        metadata = {**(original.get('metadata') or {}), 'study_status': 'aggregate_archive',
                    'experiment_layout': 'original_aggregate', 'split_verified': True,
                    'replacement_experiment_ids': [c['id'] for c in children]}
        api('PATCH', 'experiment/' + source['id'], json={'metadata': metadata,
            'description': 'ORIGINAL AGGREGATE ARCHIVE — use the verified single-model/configuration copies. All original records preserved. ' + (original.get('description') or '')})
        receipt = {'source_id': source['id'], 'source_name': source['name'], 'rows': source['rows'],
                   'children': children, 'verified': True, 'inference_calls': 0}
        receipts.append(receipt)
        receipt_path.write_text(json.dumps(receipts, indent=2) + '\n')
        print(json.dumps({'verified_source': source['name'], 'children': len(children), 'rows': source['rows']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', type=Path)
    args = parser.parse_args()
    apply(args.apply) if args.apply else prepare()
