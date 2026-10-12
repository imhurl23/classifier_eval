"""Resume frozen source requests on explicitly validated context replacements.

The original plan, requests, weights and completed decisions stay immutable.
Only serving context increases; a separate receipt records replacement bindings.
"""
import argparse
import copy
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import requests
import yaml

from data.jsonl import read_jsonl
from run import source_supplement as source
from run import pilot
from run.baseten_preflight import ALLOWLIST
from run.heldout import atomic, shutdown_baseten
from run.maintain_experiments import api
from run.pilot import load_local_service_keys
from run.split_experiments import list_experiments
from run.watch_source_completion import watch

ARMS={'tev':'tev1_native','qwen':'qwen35_4b_letters'}


def install_bindings(replacement_bindings):
    original_loader=source.load_local_service_keys
    def replacement_loader():
        original_loader()
        for arm,(_,deployment) in replacement_bindings.items():
            name='BASETEN_TEV1_DEPLOYMENT_ID' if arm=='tev1_native' else 'BASETEN_QWEN35_DEPLOYMENT_ID'
            os.environ[name]=deployment
    source.load_local_service_keys=replacement_loader
    # adapter_for reloads .env too; keep that reload on the same binding.
    pilot.load_local_service_keys=replacement_loader


def validate_config(original,replacement):
    """Allow exactly the context correction and matching two-L4 declaration."""
    expected=copy.deepcopy(original)
    command=expected['docker_server']['start_command']
    if '--max-model-len 4096' not in command:
        raise ValueError('Original context is not the diagnosed 4096 limit')
    expected['docker_server']['start_command']=command.replace('--max-model-len 4096','--max-model-len 8192')
    expected['resources']['accelerator']='L4:2'
    if replacement!=expected:
        raise ValueError('Replacement changes more than context and the GPU declaration')


def recover(directory):
    directory=Path(directory)
    summary,_,jobs=source.load_plan(directory)
    recovery=directory/'context-recovery'
    state=json.loads((directory/'baseten-execution.json').read_text())
    if state['status']=='running' or state.get('inflight'):
        raise ValueError('Recovery requires stopped, reconciled execution')
    completed={r['job_id'] for r in read_jsonl(directory/'baseten-results.jsonl') if r['status']!='api_error'}
    receipt={'plan_version':summary['version'],'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'reason':'Deployed Tev/Qwen context was 4096 despite the frozen 8192 local config',
             'remaining_job_ids':[j['job_id'] for j in jobs if j['arm'] in source.COMMON+source.NATIVE and j['job_id'] not in completed],
             'replacements':{},'status':'checking_deployments'}
    load_local_service_keys()
    headers={'Authorization':'Bearer '+os.environ['BASETEN_API_KEY']}
    started=time.monotonic()
    original_bindings=dict(ALLOWLIST)
    replacement_bindings={}
    def get(model,deployment):
        r=requests.get(f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}',headers=headers,timeout=30)
        r.raise_for_status();return r.json()
    try:
        for label,arm in ARMS.items():
            pushed=json.loads((recovery/(label+'-push.json')).read_text())
            model,old=original_bindings[arm];new=pushed['model_version_id']
            if pushed['model_id']!=model or new==old or pushed['is_draft']:
                raise ValueError('Unexpected replacement deployment identity')
            replacement_bindings[arm]=(model,new)
            config=json.loads(subprocess.check_output([str(Path(sys.executable).parent/'truss'),'model-config',
                '--model-id',model,'--deployment-id',new,'--output','json'],text=True,timeout=120))
            original=json.loads((recovery/(label+'-original-config.json')).read_text())
            validate_config(yaml.safe_load(original['raw_config']),yaml.safe_load(config['raw_config']))
            atomic(recovery/(label+'-deployed-config.json'),config)
            info=get(model,new)
            if info['is_production'] or info['is_development'] or info['instance_type_name']!=get(model,old)['instance_type_name']:
                raise ValueError('Replacement changes environment or GPU hardware')
            if info['autoscaling_settings']['min_replica']!=0 or info['autoscaling_settings']['max_replica']!=1:
                raise ValueError('Replacement replica bound differs')
            receipt['replacements'][arm]={'model_id':model,'old_deployment_id':old,'new_deployment_id':new,
                'max_model_len':8192,'weights':config['config']['weights'],'instance_type_name':info['instance_type_name']}
        ALLOWLIST.update(replacement_bindings)
        atomic(recovery/'receipt.json',receipt)
        # Push starts building immediately. Wait under a separate bounded startup
        # allowance, then return replacements to the runner's inactive precondition.
        while True:
            statuses={a:get(*binding)['status'] for a,binding in replacement_bindings.items()}
            receipt['deployment_statuses']=statuses;atomic(recovery/'receipt.json',receipt)
            if all(s in {'ACTIVE','SCALED_TO_ZERO','INACTIVE'} for s in statuses.values()):break
            if any(s in {'FAILED','UNHEALTHY'} for s in statuses.values()) or time.monotonic()-started>1200:
                raise ValueError('Context replacement readiness failed')
            time.sleep(10)
        shutdown=shutdown_baseten(set(replacement_bindings))
        if not all(x.get('verified_inactive') for x in shutdown.values()):
            raise ValueError('Replacement shutdown precondition failed')
        # Charge deployment startup to the existing cumulative compute allowance.
        state['elapsed_seconds']+=time.monotonic()-started
        atomic(directory/'baseten-execution.json',state)
        install_bindings(replacement_bindings)
        receipt['status']='resuming';atomic(recovery/'receipt.json',receipt)
        report=watch(directory,max_recoveries=3,workers_per_arm=4)
        for experiment in list_experiments():
            metadata=experiment.get('metadata') or {}
            if metadata.get('plan_version')==summary['version']:
                api('PATCH','experiment/'+experiment['id'],json={'metadata':{**metadata,
                    'serving_context_recovery':receipt['replacements'],
                    'recovery_preserves_completed_predictions':True}})
        receipt['status']=report['status'];atomic(recovery/'receipt.json',receipt)
        return report
    except BaseException as exc:
        receipt.update(status='incomplete_or_unverified',error=type(exc).__name__+': '+str(exc))
        atomic(recovery/'receipt.json',receipt)
        raise
    finally:
        ALLOWLIST.update(replacement_bindings)
        shutdown=shutdown_baseten(set(source.COMMON+source.NATIVE))
        atomic(recovery/'shutdown.json',shutdown)
        ALLOWLIST.clear();ALLOWLIST.update(original_bindings)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    if sys.platform=='darwin' and shutil.which('caffeinate'):
        subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    with (args.directory/'.context-recovery.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(recover(args.directory)),flush=True)
