"""Archive confirmed provider failures for retry; never repeat valid decisions."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil

from data.jsonl import read_jsonl
from run.heldout import atomic


def prepare_retry(directory,family='hosted'):
    prefix='' if family=='hosted' else family+'-'
    resultpath=directory/(prefix+'results.jsonl')
    statepath=directory/('execution.json' if family=='hosted' else prefix+'execution.json')
    lockpath=directory/('.execution.lock' if family=='hosted' else '.'+family+'.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(statepath.read_text());results=read_jsonl(resultpath)
        if state['status']=='running' or state.get('inflight'):raise ValueError('Run must be stopped with all inflight calls resolved')
        failed=[r for r in results if r['status']=='api_error']
        if not failed:return {'retry_jobs':0}
        archive=directory/'provider-error-history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive.mkdir(parents=True,exist_ok=False)
        shutil.copy2(resultpath,archive/resultpath.name);shutil.copy2(statepath,archive/statepath.name)
        with (archive/'failed.jsonl').open('w') as handle:
            for row in failed:handle.write(json.dumps(row,allow_nan=False)+'\n')
        retained=[r for r in results if r['status']!='api_error']
        temporary=resultpath.with_suffix('.retry.tmp')
        with temporary.open('w') as handle:
            for row in retained:handle.write(json.dumps(row,allow_nan=False)+'\n')
        temporary.replace(resultpath)
        state.update(status='partial',completed_requests=len(retained),
                     counts=dict(Counter(r['arm']+'|'+r['status'] for r in retained)))
        receipt={'archive':str(archive),'retry_jobs':len(failed),'by_arm':dict(Counter(r['arm'] for r in failed)),
                 'reason':'Confirmed api_error only; valid, refused, invalid and truncated model outputs retained without repeats'}
        state.setdefault('provider_retry_history',[]).append(receipt);atomic(statepath,state)
        return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--family',choices=('hosted','baseten','policy'),default='hosted')
    args=parser.parse_args();print(json.dumps(prepare_retry(args.directory,args.family)))
