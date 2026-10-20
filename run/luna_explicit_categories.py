"""Paired Decisions Luna diagnostic; preserves source inputs and choice order.

Run: .venv/bin/python -m run.luna_explicit_categories [--execute] [--resume]
Both fresh conditions use individual requests; only instructions differ.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import threading
import time

from adapters.openai_decisions import OpenAIDecisionsAdapter
from run.offline_review import digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/source-full/000bf1bcb6dc'
OUT = ROOT / 'data/processed/luna-explicit-categories/2026-10-14'
RULES = ROOT / 'prompts/rubrics/wildguard_explicit_categories.md'


def records(path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def variant_request(original, variant, rules):
    request = deepcopy(original)
    if variant == 'explicit_categories':
        request['questions'][0]['instructions'] += '\n\n' + rules.strip()
    elif variant != 'original_policy_repeat':
        raise ValueError(variant)
    return request


def plan():
    rules = RULES.read_text()
    samples = {r['id']: r for r in records(SOURCE / 'sample.jsonl')}
    baselines = [r for r in records(SOURCE / 'results.jsonl')
                 if r['arm'] == 'decisions_luna' and r['target'] == 'prompt_harm'
                 and samples[r['item_id']]['metadata']['source'] == 'wildguardtest']
    assert len(baselines) == len({r['item_id'] for r in baselines}) == 1681
    random.Random(20261014).shuffle(baselines)
    jobs = []
    for i, r in enumerate(baselines):
        variants = ['original_policy_repeat', 'explicit_categories']
        if i % 2:
            variants.reverse()
        for variant in variants:
            request = variant_request(r['raw']['request'], variant, rules)
            job = {'item_id': r['item_id'], 'variant': variant, 'request': request,
                   'source_gold': r['source_gold'], 'original_label': r['label'],
                   'category': samples[r['item_id']]['metadata']['wildguard_subcategory'],
                   'original_job_id': r['job_id'], 'request_sha256': digest(request)}
            job['job_id'] = digest({'variant': variant, 'item_id': r['item_id'], 'request': request})
            jobs.append(job)
    return jobs


def append(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def run(execute=False, resume=False):
    jobs = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = list(records(OUT / 'manifest.jsonl'))
        if frozen:
            if frozen != jobs:
                raise ValueError('Frozen plan changed')
        else:
            for job in jobs:
                append(OUT / 'manifest.jsonl', job)
            (OUT / 'explicit_rules.md').write_text(RULES.read_text())
        print(json.dumps({'planned': len(jobs), 'examples': len(jobs) // 2, 'execute': execute}), flush=True)
        if not execute:
            return
        previous = list(records(OUT / 'results.jsonl'))
        done = {r['job_id'] for r in previous}
        assert len(done) == len(previous) and done <= {j['job_id'] for j in jobs}
        if done and not resume:
            raise ValueError('Use --resume to preserve completed calls')
        inflight = OUT / 'inflight'
        inflight.mkdir(exist_ok=True)
        for path in inflight.iterdir():
            if path.name in done:
                path.unlink()
            else:
                raise ValueError('Unreconciled inflight request: ' + path.name)
        pending = [j for j in jobs if j['job_id'] not in done]
        local = threading.local()
        def execute_job(job):
            if not hasattr(local, 'adapter'):
                local.adapter = OpenAIDecisionsAdapter({})
            decision = local.adapter._call(job['request'])['safety']
            return {**{k: v for k, v in job.items() if k != 'request'}, **asdict(decision),
                    'model_version': local.adapter.model_version,
                    'completed_at': datetime.now(timezone.utc).isoformat()}
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0, len(pending), 4):
                if time.monotonic() - start > 1200:
                    raise RuntimeError('20-minute execution bound reached; resume remaining requests')
                wave = pending[offset:offset + 4]
                for job in wave:
                    (inflight / job['job_id']).write_text(json.dumps({'request_sha256': job['request_sha256']}))
                futures = {pool.submit(execute_job, job): job for job in wave}
                failures = []
                for future in as_completed(futures):
                    job = futures[future]
                    result = future.result()
                    assert result['raw']['request'] == job['request']
                    append(OUT / 'results.jsonl', result)
                    (inflight / job['job_id']).unlink()
                    done.add(job['job_id'])
                    if result['status'] != 'ok':
                        failures.append({'item_id': job['item_id'], 'status': result['status']})
                if failures:
                    raise RuntimeError('Provider failures saved; inspect before further calls: ' + json.dumps(failures))
                if offset % 100 == 0 or len(done) == len(jobs):
                    print(json.dumps({'completed': len(done), 'planned': len(jobs),
                                      'elapsed_s': round(time.monotonic() - start, 1)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.execute, args.resume)
