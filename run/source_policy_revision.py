"""Frozen source-policy rerun. Original manifests and results are never rewritten.

Core: source-specific Aegis interaction and WildGuard turn/refusal judgments.
Robustness: same Aegis development items and conditions as the old schema pilot
and repeatability study, plus a reversed-option diagnostic. No new held-out claim.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from adapters.guards import BasetenGuard, safeguard
from prompts.source_aligned.runtime import policy_for, VERSION
from run.explicit_categories import make_request
from run.heldout import atomic, sha
from run.matrix import matched_option_order
from run.offline_review import digest
from run.pilot import adapter_for, load_local_service_keys
from analysis.source_costs import estimate
from run.source_labels import source_gold, source_scores, usage_metrics
from run.baseten_preflight import ALLOWLIST
from data.bt_auth import load_braintrust_key
from run.experiment_groups import init_experiments

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/source-policy-revision' / VERSION
SOURCE = ROOT / 'data/processed/source-full/000bf1bcb6dc'
HOSTED = ('jev_choice', 'jev_noul', 'clef_choice', 'clef_flash_choice',
          'decisions_luna', 'luna_structured', 'claude_structured')
GPU = ('tev1_native', 'qwen35_4b_letters', 'qwen35_4b_json', 'safeguard_120b')
ARMS = HOSTED + GPU
RATES = json.loads((ROOT / 'analysis/source-pricing-2026-10-12.json').read_text())
HOSTED_CAP = 85.0
GPU_CAP = 60.0


def records(path):
    if path.exists():
        with path.open() as h:
            for line in h:
                if line.strip():
                    yield json.loads(line)


def append(path, row):
    with path.open('a') as h:
        h.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        h.flush()
        os.fsync(h.fileno())


def tasks(row):
    source = row['metadata']['source']
    if source.startswith('aegis_v2_'):
        return ('interaction_harm',)
    if source == 'wildguardtest':
        return ('prompt_harm', 'response_harm', 'response_refusal')
    return ()


def job_for(arm, row, target, *, stage='core', scheme='s2', variant='long', replicate=0, reverse=False):
    instructions, definitions = policy_for(row['metadata']['source'], target, variant, scheme)
    order = matched_option_order(row['id'], target, scheme, list(definitions))
    if reverse:
        order.reverse()
    request = make_request(arm, row, instructions, order, definitions)
    job = dict(arm=arm, item_id=row['id'], target=target, stage=stage, scheme=scheme,
               rubric='aligned_' + variant, variant=variant, replicate=replicate,
               reversed_options=reverse, instructions=instructions, definitions=definitions,
               option_order=order, request=request, request_sha256=digest(request),
               policy_version=VERSION, source_gold=source_gold(row, target))
    job['job_id'] = digest(job)
    return job


def jobs_for(arm, rows):
    for row in rows.values():
        for target in tasks(row):
            if source_gold(row, target) is None or (arm == 'jev_noul' and target != 'interaction_harm'):
                continue
            yield job_for(arm, row, target)
    if arm not in HOSTED:
        return
    # Preserve old membership; remove unsupported artificial Aegis turn tasks.
    old = ROOT / 'data/processed/study-kickoff/0ecb68a006dd/manifest.jsonl'
    for j in records(old):
        if j['arm'] != arm or j['target'] != 'interaction_harm':
            continue
        yield job_for(arm, rows[j['item_id']], j['target'], stage='robustness',
                      scheme=j['scheme'], variant=j['rubric'].removeprefix('r_'))
    old = ROOT / 'data/processed/study-repeatability/477ac56d553a/manifest.jsonl'
    for j in records(old):
        if j['arm'] != arm or j['target'] != 'interaction_harm':
            continue
        yield job_for(arm, rows[j['item_id']], j['target'], stage='repeatability', replicate=j['replicate'])
    sample_ids = sorted({j['item_id'] for j in records(ROOT / 'data/processed/study-kickoff/0ecb68a006dd/manifest.jsonl')
                         if j['target'] == 'interaction_harm'})
    if arm != 'jev_noul':
        for item in sample_ids:
            yield job_for(arm, rows[item], 'interaction_harm', stage='option_order', reverse=True)


def frozen_files():
    files = [Path(__file__).resolve(), ROOT / 'run/explicit_categories.py', ROOT / 'run/source_policy_revision_batch.py',
             ROOT / 'data/label_map.yaml', ROOT / 'prompts/label_definitions.yaml',
             ROOT / 'analysis/source-pricing-2026-10-12.json']
    files += sorted((ROOT / 'prompts/source_aligned').glob('*.md'))
    files += sorted((ROOT / 'prompts/source_aligned').glob('*.py'))
    files += sorted((ROOT / 'adapters').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise ValueError('Version already frozen; verify or choose a new policy version')
    rows = {r['id']: r for r in records(SOURCE / 'sample.jsonl')}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / 'data/raw/qwen-tokenizer/tokenizer.json'))
    counts = {}; max_context = {}; estimates = {}
    for arm in ARMS:
        folder = OUT / arm
        folder.mkdir(exist_ok=True)
        seen = set(); groups = Counter(); total_estimate = 0; maximum = 0
        with (folder / 'manifest.jsonl').open('w') as handle:
            for job in jobs_for(arm, rows):
                assert job['job_id'] not in seen
                seen.add(job['job_id']); groups[job['stage']] += 1
                n = len(tok.encode(json.dumps(job['request'], ensure_ascii=False), add_special_tokens=False).ids)
                output = job['request'].get('max_tokens', job['request'].get('max_output_tokens', 0))
                maximum = max(maximum, n + output + 256)
                if arm in GPU and n + output + 256 > 8192:
                    raise ValueError(f'Context overflow: {arm} {job["item_id"]}')
                job['input_tokens_proxy'] = n
                if arm in HOSTED:
                    total_estimate += (n * RATES[arm]['input'] + 16 * RATES[arm]['output']) / 1e6
                handle.write(json.dumps(job, ensure_ascii=False) + '\n')
        counts[arm] = dict(groups); max_context[arm] = maximum; estimates[arm] = total_estimate
    # A planning estimate, not a provider-enforced cap. Runtime ledger accounts
    # for concurrent reservations and applies a retry/unknown-usage allowance.
    if sum(estimates.values()) > HOSTED_CAP / 1.2:
        raise ValueError('Estimated hosted total leaves insufficient budget margin')
    summary = dict(version=VERSION, prepared_at=datetime.now(timezone.utc).isoformat(),
                   status='frozen', counts=counts, max_context_proxy=max_context,
                   uncached_hosted_estimate_usd=estimates, hosted_cap_usd=HOSTED_CAP,
                   gpu_cap_usd=GPU_CAP, total_cap_usd=150, max_gpu_seconds=18000,
                   policy_sources={
                       'aegis': 'https://aclanthology.org/2025.naacl-long.306.pdf',
                       'wildguard': 'https://papers.nips.cc/paper_files/paper/2024/file/0f69b4b96a46f284b726fbd70f74fb3b-Paper-Datasets_and_Benchmarks_Track.pdf'},
                   caveats=['Policy revision follows inspection of test disagreements; not preregistered or fresh held-out evidence.',
                            'Aegis detailed human category instructions are not fully published; alignment uses published taxonomy and model-policy descriptions, not a claim of identical annotation instructions.',
                            'Questionable individual labels remain unchanged; no per-example exceptions or source labels in requests.',
                            'Expanded-choice and rubric diagnostics retain small original development samples, not full-dataset robustness claims.',
                            'Unchanged XSTest and fixed-policy native references are reused with their original provenance.',
                            'Dollar estimates use October 12 snapshots and recorded usage; retries, gateway fees and provider invoices may differ.'],
                   file_sha256=frozen_files(), source_sample_sha256=sha(SOURCE / 'sample.jsonl'),
                   manifest_sha256={arm: sha(OUT / arm / 'manifest.jsonl') for arm in ARMS})
    atomic(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify():
    summary = json.loads((OUT / 'summary.json').read_text())
    if summary['file_sha256'] != frozen_files():
        raise ValueError('Frozen code or policies changed; create a new version')
    assert summary['source_sample_sha256'] == sha(SOURCE / 'sample.jsonl')
    for arm, value in summary['manifest_sha256'].items():
        assert value == sha(OUT / arm / 'manifest.jsonl'), arm
    return summary


def budget_update(job, decision=None):
    if job['arm'] not in HOSTED:
        return
    path = OUT / 'hosted-budget.json'
    with (OUT / '.budget.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {'charged_estimate_usd': 0., 'reserved': {}, 'completed': 0}
        rate = RATES[job['arm']]
        if decision is None:
            maximum_output = job['request'].get('max_tokens', job['request'].get('max_output_tokens', 0))
            reserve = 3 * (2 * job['input_tokens_proxy'] * rate['input'] + maximum_output * rate['output']) / 1e6
            if ledger['charged_estimate_usd'] + sum(ledger['reserved'].values()) + reserve > HOSTED_CAP:
                raise RuntimeError('Hosted budget bound reached')
            assert job['job_id'] not in ledger['reserved']
            ledger['reserved'][job['job_id']] = reserve
        else:
            reserve = ledger['reserved'].pop(job['job_id'])
            actual = estimate(decision.usage or {}, rate)
            # Unknown usage is charged at the full call reserve, never zero.
            ledger['charged_estimate_usd'] += reserve if actual is None else min(reserve, 1.5 * actual)
            ledger['completed'] += 1
        atomic(path, ledger)


def run(arm, resume=False, workers=4):
    verify()
    folder = OUT / arm
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = list(records(folder / 'manifest.jsonl'))
        prior = list(records(folder / 'results.jsonl'))
        done = {r['job_id'] for r in prior}
        assert len(done) == len(prior)
        if prior and not resume:
            raise ValueError('Use --resume; never automatically repeat paid calls')
        inflight = folder / 'inflight'; inflight.mkdir(exist_ok=True)
        for path in inflight.iterdir():
            if path.name in done:
                path.unlink()
            else:
                raise ValueError('Ambiguous call requires reconciliation: ' + path.name)
        rows = {r['id']: r for r in records(SOURCE / 'sample.jsonl')}
        load_local_service_keys(); load_braintrust_key()
        group = init_experiments(project='decision-interface-safety', experiment=VERSION + '__' + arm,
                    update=resume, metadata={'policy_version': VERSION, 'preregistered': False,
                    'evaluation_role': 'source_policy_revision', 'policy_mode': 'source_aligned',
                    'study_status': 'running', 'original_results_preserved': True})
        local = threading.local(); started = time.monotonic(); counts = Counter(r['status'] for r in prior)
        def work(job):
            if not hasattr(local, 'cache'):
                local.cache = {}
            key = digest(job['definitions'])
            if key not in local.cache:
                local.cache[key] = (BasetenGuard(arm, deployment_id=ALLOWLIST[arm][1]) if arm == 'safeguard_120b'
                                    else adapter_for(arm, job['definitions']))
            adapter = local.cache[key]
            row = rows[job['item_id']]
            if arm == 'safeguard_120b':
                decision = safeguard(adapter, row['input']['text'], job['instructions'], job['definitions'], job['option_order'])
            else:
                decision = adapter.classify(row['input']['text'], job['scheme'], job['instructions'], job['option_order'])
            # Save paid-call evidence even if an adapter payload drifts.
            matches = decision.raw.get('request') == job['request']
            result = {k: v for k, v in job.items() if k not in {'request', 'instructions', 'definitions', 'option_order'}}
            result.update(asdict(decision), model_version=adapter.model_version,
                          actual_request_matches=matches, completed_at=datetime.now(timezone.utc).isoformat(),
                          scores=source_scores(asdict(decision), row, job['target'], job['scheme']))
            return decision, result
        pending = [j for j in jobs if j['job_id'] not in done]
        consecutive = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                if time.monotonic() - started > 18000:
                    raise TimeoutError('Five-hour worker bound reached')
                wave = pending[offset:offset + workers]
                for job in wave:
                    budget_update(job)
                    (inflight / job['job_id']).write_text(json.dumps({'request_sha256': job['request_sha256']}))
                futures = {pool.submit(work, job): job for job in wave}
                for future in as_completed(futures):
                    job = futures[future]
                    decision, result = future.result()
                    append(folder / 'results.jsonl', result)
                    budget_update(job, decision)
                    (inflight / job['job_id']).unlink()
                    done.add(job['job_id']); counts[result['status']] += 1
                    consecutive = consecutive + 1 if result['status'] == 'api_error' else 0
                    group.log(id=result['job_id'], input=result['raw']['request'],
                              output={k: result.get(k) for k in ('label', 'probs', 'status')},
                              expected={'label': result['source_gold']}, scores=result['scores'],
                              metadata={k: result[k] for k in ('arm', 'target', 'scheme', 'rubric', 'stage', 'item_id', 'policy_version', 'replicate')},
                              metrics={'latency_ms': result['latency_ms'], **usage_metrics(result.get('usage'))})
                    if not result['actual_request_matches']:
                        raise RuntimeError('Actual payload drift; paid evidence saved, no automatic retry')
                if consecutive >= 3:
                    raise RuntimeError('Three consecutive API errors; inspect before resuming')
                if offset % (workers * 25) == 0 or len(done) == len(jobs):
                    state = dict(arm=arm, completed=len(done), planned=len(jobs), counts=dict(counts), elapsed_s=round(time.monotonic()-started, 1))
                    atomic(folder / 'progress.json', state)
                    print(json.dumps(state), flush=True)
        group.flush()
        atomic(folder / 'verification.json', dict(completed=len(done), planned=len(jobs), complete=len(done)==len(jobs),
                  actual_payloads_verified=True, experiment_urls=group.experiment_urls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.verify:
        print(json.dumps(verify(), indent=2))
    elif args.arm:
        run(args.arm, args.resume, args.workers)
    else:
        parser.error('Choose --prepare, --verify, or --arm')
