"""Finish and audit a queued Baseten source run, with bounded recovery.

Existing model decisions remain immutable during recovery. Only api_error rows
are archived for retry; remaining jobs resume under the original time budget.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import time

from data.jsonl import read_jsonl
from run.heldout import atomic,shutdown_baseten
from run.retry_source_errors import prepare_retry
from run.source_supplement import run,COMMON,NATIVE
from run.verify_source_run import verify
from run.split_experiments import list_experiments
from run.maintain_experiments import api


def watch(directory,max_recoveries=2):
    directory=Path(directory)
    with (directory/'.completion-watch.lock').open('a') as own_lock:
        fcntl.flock(own_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for attempt in range(max_recoveries+1):
                with (directory/'.baseten.lock').open('a') as execution:
                    fcntl.flock(execution.fileno(),fcntl.LOCK_EX)
                    state=json.loads((directory/'baseten-execution.json').read_text())
                    records=read_jsonl(directory/'baseten-results.jsonl')
                failed=sum(r['status']=='api_error' for r in records)
                if state['status']=='completed' and not failed:break
                if state.get('inflight'):raise ValueError('Ambiguous inflight calls need reconciliation')
                if attempt==max_recoveries:raise ValueError('Bounded recoveries exhausted; incomplete coverage retained')
                if failed:print(json.dumps(prepare_retry(directory,'baseten')),flush=True)
                run(directory,family='baseten',resume=True,workers_per_arm=8)
            summary=json.loads((directory/'summary.json').read_text())
            planned={j['job_id'] for j in read_jsonl(directory/'manifest.jsonl') if j['arm'] in COMMON+NATIVE}
            assert {r['job_id'] for r in records}==planned
            if not all(r.get('verified_inactive') for r in state.get('baseten_shutdown',{}).values()) or len(state.get('baseten_shutdown',{}))!=6:
                shutdown=shutdown_baseten(set(COMMON+NATIVE))
                assert len(shutdown)==6 and all(r.get('verified_inactive') for r in shutdown.values())
            for attempt in range(3):
                try:report=verify(directory,'baseten',remote=True);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(30)
            report.update(status='verified_complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                          gpu_shutdown_verified=True,compute_reserve_usd=summary['compute_reserve_usd'])
            atomic(directory/'completion.json',report)
            atomic(directory/'baseten-verification.json',report)
            print(json.dumps(report),flush=True)
            return report
        except Exception as exc:
            report={'status':'incomplete_or_unverified','reason':type(exc).__name__+': '+str(exc),
                    'finished_at_utc':datetime.now(timezone.utc).isoformat()}
            atomic(directory/'completion.json',report)
            version=json.loads((directory/'summary.json').read_text())['version']
            for experiment in list_experiments():
                metadata=experiment.get('metadata') or {}
                if metadata.get('plan_version')==version:
                    api('PATCH','experiment/'+experiment['id'],json={'metadata':{**metadata,
                        'study_status':'partial','verification_complete':False,'verification_error':report['reason']}})
            print(json.dumps(report),flush=True)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args();watch(args.directory)
