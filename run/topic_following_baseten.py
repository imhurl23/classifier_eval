"""Bounded restart, policy pilot, and shutdown for the existing Tev/Qwen deployments."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from data.topic_following import OUT
from run.baseten_preflight import ALLOWLIST
from run.pilot import load_local_service_keys

ROOT=Path(__file__).resolve().parents[1]


def run(output,max_seconds=900,arms=None,source_limit=4,pair_limit=6,prior_runs=None,data=OUT,workers=1):
    arms=arms or ['tev1_native','qwen35_4b_letters','qwen35_4b_json']
    if not set(arms).issubset({'tev1_native','qwen35_4b_letters','qwen35_4b_json'}):raise ValueError('Unsupported Baseten policy arm')
    deployments=[a for a in ['tev1_native','qwen35_4b_letters'] if a in arms or (a=='qwen35_4b_letters' and 'qwen35_4b_json' in arms)]
    load_local_service_keys();output.mkdir(parents=True,exist_ok=True)
    headers={'Authorization':'Bearer '+os.environ['BASETEN_API_KEY']}
    state={'started':datetime.now(timezone.utc).isoformat(),'status':'checking','max_wall_seconds':max_seconds,
           'deployments':{},'arms':arms,'source_limit':source_limit,'pair_limit':pair_limit}
    receipt=output/'deployment-lifecycle.json';activated=[];child=None;started=time.monotonic()
    def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
    def api(method,model,deployment,suffix=''):
        # Retry reads only; do not repeat ambiguous activation mutations.
        for attempt in range(3 if method=='GET' else 1):
            try:
                r=requests.request(method,f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}{suffix}',headers=headers,timeout=30)
                r.raise_for_status();return r.json()
            except requests.RequestException:
                if method!='GET' or attempt==2:raise
                time.sleep(2)
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for arm in deployments:
            model,deployment=ALLOWLIST[arm];info=api('GET',model,deployment)
            env='BASETEN_TEV1' if arm=='tev1_native' else 'BASETEN_QWEN35'
            if os.environ.get(env+'_MODEL_ID')!=model or os.environ.get(env+'_DEPLOYMENT_ID')!=deployment:
                raise ValueError('Adapter binding differs from allowlisted deployment')
            if info['id']!=deployment or info['model_id']!=model or info['status']!='INACTIVE':
                raise ValueError('Expected allowlisted inactive deployment')
            if info['instance_type_name']!='L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB':
                raise ValueError('Deployment hardware changed')
            if info['autoscaling_settings']['min_replica']!=0 or info['autoscaling_settings']['max_replica']!=1:
                raise ValueError('Expected scale to zero, maximum one replica')
            state['deployments'][arm]={k:info[k] for k in ['id','model_id','status','active_replica_count','instance_type_name']}
        state['status']='activating';save()
        for arm in deployments:
            model,deployment=ALLOWLIST[arm];activated.append((arm,model,deployment))
            state['deployments'][arm]['activate']=api('POST',model,deployment,'/activate');save()
        state['status']='waiting_for_readiness';save()
        while True:
            ready=[]
            for arm,model,deployment in activated:
                info=api('GET',model,deployment);status=info['status']
                state['deployments'][arm]['live_status']=status
                ready.append(status in {'ACTIVE','SCALED_TO_ZERO'})
                if status in {'FAILED','UNHEALTHY'}:raise RuntimeError(arm+' startup failed: '+status)
            save()
            print(json.dumps({'phase':'readiness','deployments':{a:s['live_status'] for a,s in state['deployments'].items()}}),flush=True)
            if all(ready):break
            if time.monotonic()-started>max_seconds-180:raise TimeoutError('Startup deadline reached')
            time.sleep(10)
        state['status']='running';save()
        command=[str(ROOT/'.venv/bin/python'),'-m','run.topic_following','--arms',*state['arms'],'--output',str(output),
                 '--data',str(data),'--workers',str(workers),'--source-limit',str(source_limit),'--pair-limit',str(pair_limit),'--execute']
        if prior_runs:command += ['--reuse-run',*[str(p) for p in prior_runs]]
        child=subprocess.Popen(command,cwd=ROOT)
        while child.poll() is None:
            if time.monotonic()-started>max_seconds:raise TimeoutError('Pilot deadline reached')
            time.sleep(1)
        state['exit_code']=child.returncode
        state['status']='completed' if child.returncode==0 else 'failed'
    except BaseException as exc:
        state['status']='stopped';state['error']=type(exc).__name__+': '+str(exc)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for arm,model,deployment in reversed(activated):
            try:state['deployments'][arm]['deactivate']=api('POST',model,deployment,'/deactivate')
            except Exception as exc:state['deployments'][arm]['shutdown_error']=type(exc).__name__+': '+str(exc)
        for arm,model,deployment in activated:
            try:
                info=api('GET',model,deployment)
                state['deployments'][arm]['final_status']=info['status']
                state['deployments'][arm]['final_replicas']=info['active_replica_count']
            except Exception as exc:state['deployments'][arm]['verification_error']=type(exc).__name__+': '+str(exc)
        state['finished']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(state,indent=2),flush=True)
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'pilot-baseten')
    p.add_argument('--arms',nargs='+',default=['tev1_native','qwen35_4b_letters','qwen35_4b_json'])
    p.add_argument('--source-limit',type=int,default=4);p.add_argument('--pair-limit',type=int,default=6)
    p.add_argument('--reuse-run',type=Path,nargs='*',default=[])
    p.add_argument('--data',type=Path,default=OUT);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--max-seconds',type=int,default=900);args=p.parse_args()
    if not 180<=args.max_seconds<=900:raise ValueError('Pilot wall limit must be 180-900 seconds')
    if args.source_limit<0 or args.pair_limit<0:raise ValueError('Invalid case limits')
    result=run(args.output,args.max_seconds,args.arms,args.source_limit,args.pair_limit,args.reuse_run,args.data,args.workers)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
