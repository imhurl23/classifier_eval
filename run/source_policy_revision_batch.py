"""Supervise isolated per-arm workers; optionally bound Baseten GPU lifecycle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from run.source_policy_revision import ARMS, GPU, HOSTED, OUT, ROOT, verify
from run.baseten_preflight import ALLOWLIST
from run.pilot import load_local_service_keys


def run(arms, baseten=False, max_seconds=18000, resume=False, workers=4):
    verify()
    if not 0 < max_seconds <= 18000:
        raise ValueError('Five-hour lifecycle bound')
    if baseten and not set(arms) <= set(GPU):
        raise ValueError('Baseten lifecycle only supports GPU arms')
    if not baseten and set(arms) & set(GPU):
        raise ValueError('GPU arms require lifecycle supervision')
    if not arms or len(arms) != len(set(arms)) or not set(arms) <= set(ARMS):
        raise ValueError('Invalid arms')
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    receipt = OUT / ('lifecycle-' + '-'.join(arms) + '.json')
    state = {'arms': arms, 'started': datetime.now(timezone.utc).isoformat(),
             'max_seconds': max_seconds, 'status': 'checking', 'deployments': {}, 'workers': {}}
    active, children, handles = [], {}, []
    started = time.monotonic()
    def save():
        receipt.write_text(json.dumps(state, indent=2) + '\n')
    def api(method, arm, suffix=''):
        model, deployment = ALLOWLIST[arm]
        for attempt in range(3 if method == 'GET' else 1):
            try:
                response = requests.request(method,
                    f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}{suffix}',
                    headers={'Authorization': 'Bearer ' + os.environ['BASETEN_API_KEY']}, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if method != 'GET' or attempt == 2:
                    raise
                time.sleep(2)
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if baseten:
            load_local_service_keys()
            deployments = sorted({('qwen35_4b_letters' if a == 'qwen35_4b_json' else a) for a in arms})
            for arm in deployments:
                info = api('GET', arm)
                model, deployment = ALLOWLIST[arm]
                env = {'tev1_native': 'BASETEN_TEV1', 'qwen35_4b_letters': 'BASETEN_QWEN35',
                       'safeguard_120b': 'BASETEN_SAFEGUARD', 'granite_41_8b': 'BASETEN_GRANITE'}[arm]
                assert os.environ[env + '_MODEL_ID'] == model
                if arm in {'tev1_native', 'qwen35_4b_letters'}:
                    assert os.environ[env + '_DEPLOYMENT_ID'] == deployment
                assert info['model_id'] == model and info['id'] == deployment
                assert info['status'] == 'INACTIVE' and info['active_replica_count'] == 0
                hardware = ('H100 - 1 H100 GPU, 80 GiB VRAM' if arm == 'safeguard_120b' else
                            'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB'
                            if arm in {'tev1_native', 'qwen35_4b_letters'} else
                            'L4:4x16 - 1 L4 GPU, 24 GiB video memory, 4 vCPU, 16 GiB')
                assert info['instance_type_name'] == hardware
                assert info['autoscaling_settings']['min_replica'] == 0
                assert info['autoscaling_settings']['max_replica'] == 1
                state['deployments'][arm] = {k: info[k] for k in ['id', 'model_id', 'status', 'active_replica_count', 'instance_type_name']}
            save()
            for arm in deployments:
                active.append(arm)  # Cleanup also covers an ambiguous activation response.
                state['deployments'][arm]['activate'] = api('POST', arm, '/activate')
                save()
            while True:
                ready = []
                for arm in active:
                    info = api('GET', arm)
                    state['deployments'][arm]['live_status'] = info['status']
                    ready.append(info['status'] in {'ACTIVE', 'SCALED_TO_ZERO'})
                    if info['status'] in {'FAILED', 'UNHEALTHY'}:
                        raise RuntimeError(arm + ' startup failed')
                state['status'] = 'readiness'
                save()
                print(json.dumps({'phase': 'readiness', 'deployments': {a: d.get('live_status') for a, d in state['deployments'].items()}}), flush=True)
                if all(ready):
                    break
                if time.monotonic() - started > min(1200, max_seconds - 180):
                    raise TimeoutError('Deployment startup deadline')
                time.sleep(10)
        for arm in arms:
            logfile = logs / (arm + '.log')
            handle = logfile.open('a')
            handles.append(handle)
            python = ROOT / ('.venv-policylm/bin/python' if arm == 'policylm_17b_prompt' else '.venv/bin/python')
            command = [str(python), '-m', 'run.source_policy_revision', '--arm', arm,
                       '--workers', '1' if arm == 'policylm_17b_prompt' else str(workers)]
            if resume:
                command.append('--resume')
            children[arm] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            state['workers'][arm] = {'pid': children[arm].pid, 'log': str(logfile)}
        state['status'] = 'running'
        save()
        last = 0
        while any(child.poll() is None for child in children.values()):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError('Batch wall-time bound reached')
            if time.monotonic() - last > 25:
                for arm, child in children.items():
                    state['workers'][arm]['exit_code'] = child.poll()
                save()
                print(json.dumps({'phase': 'running', 'elapsed_s': round(time.monotonic() - started),
                                  'exit_codes': {a: c.poll() for a, c in children.items()}}), flush=True)
                last = time.monotonic()
            time.sleep(1)
        for arm, child in children.items():
            state['workers'][arm]['exit_code'] = child.returncode
        state['status'] = 'completed' if all(c.returncode == 0 for c in children.values()) else 'failed'
    except BaseException as exc:
        state['status'], state['error'] = 'stopped', type(exc).__name__ + ': ' + str(exc)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for arm in reversed(active):
            try:
                state['deployments'][arm]['deactivate'] = api('POST', arm, '/deactivate')
            except Exception as exc:
                state['deployments'][arm]['shutdown_error'] = str(exc)
        for arm in active:
            try:
                for _ in range(6):
                    info = api('GET', arm)
                    if info['status'] == 'INACTIVE' and info['active_replica_count'] == 0:
                        break
                    time.sleep(5)
                state['deployments'][arm]['final_status'] = info['status']
                state['deployments'][arm]['final_replicas'] = info['active_replica_count']
            except Exception as exc:
                state['deployments'][arm]['verification_error'] = str(exc)
        for handle in handles:
            handle.close()
        state['finished'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps(state, indent=2), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', default=[a for a in HOSTED if a != 'decisions_luna'])
    parser.add_argument('--baseten', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=18000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 180 <= args.max_seconds <= 18000:
        raise ValueError('Invalid wall bound')
    result = run(args.arms, args.baseten, args.max_seconds, args.resume, args.workers)
    if result['status'] != 'completed':
        raise SystemExit(1)
