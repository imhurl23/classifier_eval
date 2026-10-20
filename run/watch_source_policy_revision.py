"""Finish offline analysis only after all paid workers and GPU cleanup verify."""
from datetime import datetime, timezone
import json
import subprocess
import time

from run.source_policy_revision import ROOT, OUT, ARMS, verify
from run.heldout import atomic


def main():
    verify()
    started = time.monotonic()
    receipt = OUT / 'completion.json'
    while time.monotonic() - started < 6 * 3600:
        finished = []
        for arm in ARMS:
            path = OUT / arm / 'verification.json'
            if path.exists() and json.loads(path.read_text()).get('complete'):
                finished.append(arm)
        lifecycles = [json.loads(p.read_text()) for p in OUT.glob('lifecycle-*.json')]
        errors = [s for s in lifecycles if s.get('status') in {'failed', 'stopped'}]
        if errors:
            atomic(receipt, {'status': 'needs_review', 'completed_arms': finished,
                            'failed_batches': [s['arms'] for s in errors],
                            'reason': 'Paid calls are not automatically repeated; inspect worker and lifecycle logs.'})
            raise SystemExit(1)
        cleanup = any(s.get('deployments') and s.get('status') == 'completed' and all(
            d.get('final_status') == 'INACTIVE' and d.get('final_replicas') == 0
            for d in s['deployments'].values()) for s in lifecycles)
        state = {'status': 'waiting', 'completed_arms': finished, 'planned_arms': len(ARMS),
                 'gpu_shutdown_verified': cleanup, 'updated_at': datetime.now(timezone.utc).isoformat()}
        atomic(receipt, state)
        if len(finished) == len(ARMS) and cleanup:
            break
        time.sleep(15)
    else:
        raise TimeoutError('Six-hour offline completion watcher deadline')
    with (OUT / 'analysis.log').open('a') as log:
        result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'analysis.source_policy_revision'],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    state['status'] = 'complete' if result.returncode == 0 else 'analysis_failed'
    state['analysis_exit_code'] = result.returncode
    state['analysis_directory'] = str(ROOT / 'analysis/output/source-policy-revision-2026-10-19')
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic(receipt, state)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
