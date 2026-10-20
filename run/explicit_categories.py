"""Matched original/explicit category policies across every custom-policy arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import threading
import time

from adapters import claude, luna, openai_decisions, qwen_json, tev1, jev, clef
from adapters.guards import BasetenGuard, safeguard, granite
from adapters.policylm import make_prompt_request, PolicyLMPromptAdapter
from prompts.runtime import options_for, instructions_for
from run.baseten_preflight import ALLOWLIST
from run.luna_explicit_categories import ROOT, SOURCE, RULES, records, append, OUT as LUNA_RUN
from run.matrix import matched_option_order
from run.offline_review import CaptureGuard, digest
from run.pilot import adapter_for, load_local_service_keys

HOSTED = ['jev_choice', 'jev_noul', 'clef_choice', 'clef_flash_choice',
          'luna_structured', 'claude_structured', 'decisions_luna']
GPU = ['tev1_native', 'qwen35_4b_letters', 'qwen35_4b_json', 'safeguard_120b', 'granite_41_8b']
POLICY = 'policylm_17b_prompt'
ARMS = HOSTED + GPU + [POLICY]
OUT = ROOT / 'data/processed/explicit-categories/2026-10-14'


def make_request(arm, row, instructions, order, definitions):
    text = row['input']['text']
    if arm in {'jev_choice', 'clef_choice', 'clef_flash_choice'}:
        model = {'jev_choice': 'jev-1.13.0', 'clef_choice': 'clef', 'clef_flash_choice': 'clef-flash'}[arm]
        return clef.make_choice_request(text, instructions, order, definitions, model)
    if arm == 'jev_noul':
        return jev.make_noul_request(text, instructions, definitions)
    if arm == 'decisions_luna':
        return openai_decisions.make_choice_request(text, instructions, order, definitions)
    if arm == 'luna_structured':
        return luna.make_request(text, instructions, order, definitions)
    if arm == 'claude_structured':
        return claude.make_request(text, instructions, order, definitions)
    if arm in tev1.MODEL_INFO:
        return tev1.make_request(text, instructions, order, definitions, tev1.MODEL_INFO[arm][0])
    if arm == 'qwen35_4b_json':
        return qwen_json.make_request(text, instructions, order, definitions)
    if arm == POLICY:
        return make_prompt_request(row['input']['prompt'], instructions, definitions)
    capture = CaptureGuard(arm)
    if arm == 'safeguard_120b':
        safeguard(capture, text, instructions, definitions, order)
    elif arm == 'granite_41_8b':
        granite(capture, row['input']['prompt'], row['input']['response'],
                instructions + '\nUnsafe criterion: ' + definitions['unsafe'], 'prompt_harm')
    else:
        raise ValueError(arm)
    assert len(capture.requests) == 1
    return capture.requests[0]


def plan(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    rows = [r for r in records(SOURCE / 'sample.jsonl') if r['metadata']['source'] == 'wildguardtest']
    assert len(rows) == len({r['id'] for r in rows}) == 1681
    random.Random(20261014).shuffle(rows)
    labels, definitions = options_for('s2', 'prompt_harm_s2')
    base = instructions_for('prompt_harm_s2')
    rules = RULES.read_text().strip()
    jobs = []
    for i, row in enumerate(rows):
        order = matched_option_order(row['id'], 'prompt_harm', 's2', labels)
        variants = ['original_policy_repeat', 'explicit_categories']
        if i % 2:
            variants.reverse()
        for variant in variants:
            instructions = base + ('\n\n' + rules if variant == 'explicit_categories' else '')
            request = make_request(arm, row, instructions, order, definitions)
            job = {'arm': arm, 'item_id': row['id'], 'variant': variant,
                   'category': row['metadata']['wildguard_subcategory'],
                   'source_gold': row['expected']['prompt_s2'], 'instructions': instructions,
                   'option_order': order, 'request': request, 'request_sha256': digest(request),
                   'input_context': 'user_only' if arm == POLICY else 'original_pair'}
            job['job_id'] = digest({'arm': arm, 'item_id': row['id'], 'variant': variant, 'request': request})
            jobs.append(job)
    return rows, definitions, jobs


def run(arm, execute=False, resume=False, workers=4):
    if not 1 <= workers <= 8 or (arm == POLICY and workers != 1):
        raise ValueError('Invalid worker count')
    rows, definitions, jobs = plan(arm)
    samples = {r['id']: r for r in rows}
    out = OUT / arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior_plan = list(records(out / 'manifest.jsonl'))
        if prior_plan:
            assert prior_plan == jobs, 'Frozen plan changed'
        else:
            for job in jobs:
                append(out / 'manifest.jsonl', job)
            (out / 'explicit_rules.md').write_text(RULES.read_text())
        print(json.dumps({'arm': arm, 'planned': len(jobs), 'execute': execute}), flush=True)
        if not execute:
            return
        done_rows = list(records(out / 'results.jsonl'))
        done = {r['job_id'] for r in done_rows}
        assert len(done) == len(done_rows) and done <= {j['job_id'] for j in jobs}
        if done and not resume:
            raise ValueError('Use --resume')
        inflight = out / 'inflight'
        inflight.mkdir(exist_ok=True)
        for path in inflight.iterdir():
            if path.name in done:
                path.unlink()
            else:
                raise ValueError('Unreconciled inflight call: ' + path.name)
        # Exact payload reuse only; retain the original raw evidence and timestamps.
        if arm == 'decisions_luna':
            prior = {(r['item_id'], r['variant']): r for r in records(LUNA_RUN / 'results.jsonl')}
            for job in jobs:
                if job['job_id'] in done:
                    continue
                old = prior.get((job['item_id'], job['variant']))
                if old and old['status'] == 'ok' and old['raw']['request'] == job['request']:
                    result = {**old, **{k: v for k, v in job.items() if k not in {'request', 'instructions', 'option_order'}},
                              'reused_from': str(LUNA_RUN), 'reused_job_id': old['job_id']}
                    append(out / 'results.jsonl', result)
                    done.add(job['job_id'])
        pending = [j for j in jobs if j['job_id'] not in done]
        if not pending:
            print(json.dumps({'arm': arm, 'completed': len(done), 'new_calls': 0}), flush=True)
            return
        load_local_service_keys()
        local = threading.local()
        def work(job):
            if not hasattr(local, 'adapter'):
                if arm in {'safeguard_120b', 'granite_41_8b'}:
                    local.adapter = BasetenGuard(arm, deployment_id=ALLOWLIST[arm][1])
                elif arm == POLICY:
                    local.adapter = PolicyLMPromptAdapter(definitions)
                else:
                    local.adapter = adapter_for(arm, definitions)
            adapter = local.adapter
            row = samples[job['item_id']]
            instructions, order = job['instructions'], job['option_order']
            if arm == 'safeguard_120b':
                decision = safeguard(adapter, row['input']['text'], instructions, definitions, order)
            elif arm == 'granite_41_8b':
                decision = granite(adapter, row['input']['prompt'], row['input']['response'],
                                   instructions + '\nUnsafe criterion: ' + definitions['unsafe'], 'prompt_harm')
            elif arm == POLICY:
                decision = adapter.classify_prompt(row['input']['prompt'], instructions)
            else:
                decision = adapter.classify(row['input']['text'], 's2', instructions, order)
            assert decision.raw['request'] == job['request'], 'Actual request differs from frozen payload'
            return {**{k: v for k, v in job.items() if k not in {'request', 'instructions', 'option_order'}},
                    **asdict(decision), 'model_version': adapter.model_version,
                    'completed_at': datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        consecutive = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                if time.monotonic() - started > 7200:
                    raise RuntimeError('Two-hour arm execution bound reached')
                wave = pending[offset:offset + workers]
                for job in wave:
                    (inflight / job['job_id']).write_text(json.dumps({'request_sha256': job['request_sha256']}))
                futures = {pool.submit(work, job): job for job in wave}
                for future in as_completed(futures):
                    job = futures[future]
                    result = future.result()
                    append(out / 'results.jsonl', result)
                    done.add(job['job_id'])
                    (inflight / job['job_id']).unlink()
                    consecutive = consecutive + 1 if result['status'] == 'api_error' else 0
                if consecutive >= 3:
                    raise RuntimeError('Three consecutive API failures; inspect saved evidence')
                if offset % (workers * 25) == 0 or len(done) == len(jobs):
                    print(json.dumps({'arm': arm, 'completed': len(done), 'planned': len(jobs),
                                      'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.arm, args.execute, args.resume, args.workers)
