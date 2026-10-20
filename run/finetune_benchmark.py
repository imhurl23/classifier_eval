"""Evaluate the two existing custom LoRA checkpoints; never retrain or substitute Tev1.

Prepare a frozen manifest, audit training/dev overlap, then execute with Loops.
Results are checkpointed after every request; ambiguous calls require inspection.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import threading
import time

from adapters.tev1 import make_request
from prompts.runtime import instructions_for, options_for
from prompts.source_aligned.runtime import policy_for, VERSION
from run.heldout import atomic, sha
from run.matrix import matched_option_order
from run.offline_review import digest
from run.pilot import load_local_service_keys
from run.source_labels import source_gold, source_scores
from training.build_safety_sft import normalized, Renderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/custom-finetune-benchmark/2026-10-19'
SOURCE = ROOT / 'data/processed/source-full/000bf1bcb6dc/sample.jsonl'
MODELS = {
    'safety_tev': ('safety-tev-mini', 'safety-mini'),
    'cc_repro': ('cc-repro-mini', 'cc-mini'),
}


def records(path):
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def text_keys(text):
    """Compare full states and individual turns, with punctuation/case normalized."""
    parts = [text] + re.split(r'(?:^|\n\s*\n)(?:User|Assistant):\s*\n', text)
    return {normalized(p) for p in parts if normalized(p)}


def exposure(arm):
    name, folder = MODELS[arm]
    run_path = ROOT / f'training/data/{name}-run.json'
    trained = json.loads(run_path.read_text())
    base = ROOT / 'training/data' / folder
    # Fail closed if current exports are not the exact data used for this checkpoint.
    for file, expected in trained['data_manifest']['files'].items():
        if sha(base / file) != expected:
            raise ValueError('Training artifact changed: ' + str(base / file))
    texts, source_ids = set(), set()
    for row in records(base / 'records.jsonl'):
        texts.update(text_keys(row['state']))
        source_id = row.get('provenance', {}).get('aegis_id')
        if source_id:
            source_ids.add(str(source_id))
    return trained, texts, source_ids


def overlap(row, texts, source_ids):
    reasons = []
    if str(row['metadata'].get('source_id', '')) in source_ids:
        reasons.append('source_id_in_training_or_validation')
    for field in ('text', 'prompt', 'response'):
        value = row['input'].get(field)
        if value and text_keys(value) & texts:
            reasons.append(field + '_matches_training_or_validation')
    return reasons


def tasks(row):
    source = row['metadata']['source']
    if source.startswith('aegis_v2_'):
        return ('interaction_harm',)
    if source == 'wildguardtest':
        return ('prompt_harm', 'response_harm', 'response_refusal')
    if source == 'xstest':
        return ('prompt_harm',)
    raise ValueError(source)


def prepare():
    if (OUT / 'summary.json').exists():
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    renderer = Renderer()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / 'data/raw/qwen-tokenizer/tokenizer.json'))
    samples = list(records(SOURCE))
    summary = {'policy_version': VERSION, 'source_sha256': sha(SOURCE), 'items': len(samples),
               'arms': {}, 'created_at': datetime.now(timezone.utc).isoformat(),
               'evaluation_role': 'exploratory_existing_custom_checkpoints',
               'exclusion_rule': 'Per checkpoint: exclude normalized full-text or individual-turn matches and Aegis source-ID matches to all training AND validation records.',
               'notes': ['No retraining; both existing checkpoints evaluated on all supported benchmark tasks.',
                         'Training includes prompt-only examples; paired and refusal tasks may be out of training distribution.',
                         'Published source labels unchanged. XSTest uses original general request-harm instructions.',
                         'No fresh held-out claim; existing inspected benchmarks.',
                         'WildGuard interaction OR can be derived from the two physical turn judgments; it is not a separately asked task.']}
    for arm in MODELS:
        trained, texts, ids = exposure(arm)
        folder = OUT / arm
        folder.mkdir(exist_ok=True)
        excluded, counts, jobs = [], Counter(), []
        for row in samples:
            reasons = overlap(row, texts, ids)
            if reasons:
                excluded.append({'item_id': row['id'], 'source': row['metadata']['source'], 'reasons': reasons})
                continue
            for target in tasks(row):
                if source_gold(row, target) is None:
                    continue
                if row['metadata']['source'] == 'xstest':
                    instructions = instructions_for('prompt_harm_s2')
                    _, definitions = options_for('s2', 'prompt_harm_s2')
                    policy_version = 'original_unchanged_xstest'
                else:
                    instructions, definitions = policy_for(row['metadata']['source'], target)
                    policy_version = VERSION
                order = matched_option_order(row['id'], target, 's2', list(definitions))
                request = make_request(row['input']['text'], instructions, order, definitions,
                                       trained['sampler_weights_path'])
                # Use exactly the pinned non-thinking chat template used for training.
                prompt = renderer.prompt(request['messages'] + [{'role': 'assistant', 'content': ''}])
                tokens = tok.encode(prompt, add_special_tokens=False).ids
                if len(tokens) + 8 > 8192:
                    raise ValueError('Context overflow; do not truncate: ' + row['id'])
                job = {'arm': arm, 'item_id': row['id'], 'target': target, 'scheme': 's2',
                       'source': row['metadata']['source'], 'unit': row['input']['unit'],
                       'rubric': 'aligned_long' if policy_version == VERSION else 'prompt_harm_s2',
                       'policy_version': policy_version, 'checkpoint': trained['sampler_weights_path'],
                       'option_order': order, 'request': request, 'prompt': prompt,
                       'input_tokens': len(tokens), 'source_gold': source_gold(row, target)}
                job['job_id'] = digest(job)
                jobs.append(job)
                counts[row['metadata']['source'] + ':' + target] += 1
        manifest = folder / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs))
        atomic(folder / 'exclusions.json', excluded)
        summary['arms'][arm] = {'checkpoint': trained['sampler_weights_path'],
            'training_record': MODELS[arm][0] + '-run.json', 'run_id': trained['run_id'],
            'training_run_sha256': sha(ROOT / f'training/data/{MODELS[arm][0]}-run.json'),
            'excluded_items': len(excluded), 'eligible_items': len(samples) - len(excluded),
            'requests': len(jobs), 'counts': dict(counts), 'manifest_sha256': sha(manifest),
            'exclusions_sha256': sha(folder / 'exclusions.json'),
            'max_input_tokens': max(j['input_tokens'] for j in jobs)}
    summary['file_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in
        [Path(__file__).resolve(), ROOT / 'adapters/tev1.py', ROOT / 'prompts/runtime.py',
         ROOT / 'training/build_safety_sft.py', ROOT / 'run/source_labels.py',
         ROOT / 'scorers/scorers.py', ROOT / 'data/raw/qwen-tokenizer/tokenizer.json',
         ROOT / 'data/raw/qwen-tokenizer/chat_template.jinja'] +
        sorted((ROOT / 'prompts/source_aligned').glob('*.md')) +
        sorted((ROOT / 'prompts/source_aligned').glob('*.py'))}
    atomic(OUT / 'summary.json', summary)
    return summary


def verify():
    summary = json.loads((OUT / 'summary.json').read_text())
    assert sha(SOURCE) == summary['source_sha256']
    for file, expected in summary['file_sha256'].items():
        assert sha(ROOT / file) == expected, file
    for arm, meta in summary['arms'].items():
        exposure(arm)
        assert sha(ROOT / f"training/data/{meta['training_record']}") == meta['training_run_sha256']
        assert sha(OUT / arm / 'manifest.jsonl') == meta['manifest_sha256']
        assert sha(OUT / arm / 'exclusions.json') == meta['exclusions_sha256']
    return summary


def execute(arm, workers=8, max_seconds=14400):
    """Bound sampler lifetime, persist before/after each paid call, verify shutdown."""
    from baseten.loops import ServiceClient, ModelInput, SamplingParams
    from tokenizers import Tokenizer
    import requests
    load_local_service_keys()
    summary = verify()
    folder = OUT / arm
    lock = (folder / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = list(records(folder / 'manifest.jsonl'))
    done = {r['job_id']: r for r in records(folder / 'results.jsonl')}
    assert len(done) == sum(1 for _ in records(folder / 'results.jsonl'))
    remaining = [j for j in jobs if j['job_id'] not in done]
    inflight = folder / 'inflight'
    inflight.mkdir(exist_ok=True)
    unresolved = [p for p in inflight.glob('*.json') if p.stem not in done]
    if unresolved:
        raise ValueError('Inspect ambiguous requests before resuming: ' + str(unresolved))
    if not remaining:
        return
    state = {'arm': arm, 'status': 'starting', 'checkpoint': summary['arms'][arm]['checkpoint'],
             'max_seconds': max_seconds, 'started_at': datetime.now(timezone.utc).isoformat()}
    receipt = folder / 'lifecycle.json'
    atomic(receipt, state)
    class RecordedService(ServiceClient):
        def _make_sampler_deployment(self, sampler):
            result = super()._make_sampler_deployment(sampler)
            state['deployment'] = {'model_id': result.model_id, 'deployment_id': result.deployment_id}
            atomic(receipt, state)
            return result
    started = time.monotonic()
    mutex = threading.Lock()
    sampler = None
    tok = Tokenizer.from_file(str(ROOT / 'data/raw/qwen-tokenizer/tokenizer.json'))
    rows = {r['id']: r for r in records(SOURCE)}
    try:
        service = RecordedService()
        state['session_id'] = service.session_id
        atomic(receipt, state)
        sampler = service.create_sampling_client(model_path=state['checkpoint'], max_seq_len=8192,
                                                  timeout=120, ready_timeout=900)
        state['status'] = 'running'
        atomic(receipt, state)
        def work(job):
            if time.monotonic() - started > max_seconds - 180:
                raise TimeoutError('Sampler lifetime bound reached')
            marker = inflight / (job['job_id'] + '.json')
            atomic(marker, {'job_id': job['job_id'], 'started_at': datetime.now(timezone.utc).isoformat()})
            tokens = tok.encode(job['prompt'], add_special_tokens=False).ids
            t = time.monotonic()
            response = sampler.sample(prompt=ModelInput.from_ints(tokens), num_samples=1,
                                      sampling_params=SamplingParams(max_tokens=8, temperature=0, seed=42))
            output = tok.decode(response.sequences[0].tokens, skip_special_tokens=True).strip()
            label = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWX', job['option_order'])).get(output)
            result = {k: v for k, v in job.items() if k not in {'prompt', 'request'}}
            result.update(label=label, status='ok' if label else 'invalid_output', probs=None,
                model_version=job['checkpoint'], latency_ms=(time.monotonic()-t)*1000,
                usage={'prompt_tokens': len(tokens), 'completion_tokens': len(response.sequences[0].tokens)},
                raw={'request_sha256': digest(job['request']), 'output': output,
                     'response': response.model_dump(mode='json')},
                completed_at=datetime.now(timezone.utc).isoformat())
            result['scores'] = source_scores(result, rows[job['item_id']], job['target'], 's2')
            with mutex:
                append(folder / 'results.jsonl', result)
                done[job['job_id']] = result
                marker.unlink()
                if len(done) % 100 == 0:
                    progress = {'completed': len(done), 'planned': len(jobs),
                                'elapsed_seconds': round(time.monotonic()-started),
                                'statuses': dict(Counter(r['status'] for r in done.values()))}
                    atomic(folder / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            return result
        # Submit bounded batches so a failure does not leave thousands of queued requests.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(remaining), workers):
                futures = [pool.submit(work, j) for j in remaining[start:start+workers]]
                for future in as_completed(futures):
                    future.result()
        state['status'] = 'completed'
        atomic(folder / 'verification.json', {'complete': len(done) == len(jobs),
            'planned': len(jobs), 'completed': len(done), 'statuses': dict(Counter(r['status'] for r in done.values())),
            'results_sha256': sha(folder / 'results.jsonl')})
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        if sampler:
            sampler.close()
        deployment = state.get('deployment')
        if deployment:
            url = f"https://api.baseten.co/v1/models/{deployment['model_id']}/deployments/{deployment['deployment_id']}"
            headers = {'Authorization': 'Bearer ' + os.environ['BASETEN_API_KEY']}
            response = requests.post(url + '/deactivate', headers=headers, timeout=30)
            state['shutdown_http_status'] = response.status_code
            for _ in range(24):
                info = requests.get(url, headers=headers, timeout=30)
                info.raise_for_status()
                payload = info.json()
                state['shutdown_status'] = payload.get('status')
                state['shutdown_replicas'] = payload.get('active_replica_count')
                atomic(receipt, state)
                if payload.get('status') == 'INACTIVE' and payload.get('active_replica_count') == 0:
                    state['shutdown_verified'] = True
                    break
                time.sleep(5)
        state['elapsed_seconds'] = round(time.monotonic()-started)
        atomic(receipt, state)
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--arm', choices=list(MODELS))
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
    elif args.arm:
        if not 1 <= args.workers <= 16:
            raise ValueError('Workers must be between 1 and 16')
        execute(args.arm, args.workers)
    else:
        ap.error('Use --prepare or --arm')
