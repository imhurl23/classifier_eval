"""Bounded restart/pilot/shutdown of only the six allowlisted study deployments."""
import argparse,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import requests
from run.pilot import load_local_service_keys
ROOT=Path(__file__).resolve().parents[1]
ALLOWLIST={'tev1_native':('q0pg2dy3','wxek2l6'),'qwen35_4b_letters':('wgvmr76w','wxek2lz'),
           'safeguard_120b':('wl5m85e3','31r6mlg'),'granite_41_8b':('qk5pm5lq','qvxk5mx'),
           'qwen3guard_8b':('wozlgzz3','qrmn8p9'),'nemotron_guard':('wp52658q','qj7g9jl')}

def main(arms=None):
    selected=set(arms or ALLOWLIST)
    if not selected.issubset(ALLOWLIST): raise ValueError("Unsupported study deployment")
    chosen={a:v for a,v in ALLOWLIST.items() if a in selected}
    load_local_service_keys()
    headers={'Authorization':'Bearer '+os.environ['BASETEN_API_KEY']}
    out=ROOT/'data/processed/baseten-preflight'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    state={'status':'preparing','authorization':'User: Also include the Baseten models','max_wall_minutes':30,
           'compute_reserve_usd':25,'pricing_url':'https://www.baseten.co/pricing/','deployments':{},'pilots':{}}
    def save(): (out/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
    def api(method,model,deployment,suffix=''):
        r=requests.request(method,f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}{suffix}',headers=headers,timeout=30)
        r.raise_for_status();return r.json()
    for arm,(model,deployment) in chosen.items():
        info=api('GET',model,deployment)
        if info['model_id']!=model or info['id']!=deployment or info['status']!='INACTIVE':
            raise ValueError('Allowlisted deployment identity/inactive status mismatch')
        expected_gpu = 'H100 - 1 H100 GPU, 80 GiB VRAM' if arm == 'safeguard_120b' else 'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB' if arm in {'tev1_native','qwen35_4b_letters'} else 'L4:4x16 - 1 L4 GPU, 24 GiB video memory, 4 vCPU, 16 GiB'
        if info['instance_type_name'] != expected_gpu:
            raise ValueError('Selected hardware changed; recompute reserve before activation')
        settings=info['autoscaling_settings']
        if settings['min_replica']!=0 or settings['max_replica']!=1:
            raise ValueError('Expected scale to zero, maximum one replica')
        state['deployments'][arm]={k:info[k] for k in ('id','model_id','status','active_replica_count','instance_type_name','autoscaling_settings')}
    # 7 L4 GPUs + one H100, with startup/idle included, using a 2x rate buffer.
    estimate=30*(7*.01414+.10833)*2
    prior_preflights = len(list((ROOT/'data/processed/baseten-preflight').glob('*/execution.json')))
    cumulative_reserve = 811.03392 + 25 * (prior_preflights + 1)
    if estimate>25 or cumulative_reserve>1000: raise ValueError('Study reserve exceeded')
    state['cumulative_study_planning_reserve_usd'] = cumulative_reserve
    state['estimated_compute_upper_planning_usd']=estimate
    state['started_at_utc']=datetime.now(timezone.utc).isoformat()
    save()
    started=time.monotonic();active=[];children=[]
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        state['status']='activating';save()
        for arm,(model,deployment) in chosen.items():
            active.append((arm,model,deployment))
            state['deployments'][arm]['activate']=api('POST',model,deployment,'/activate');save()
        state['status']='waiting_for_readiness';save()
        while True:
            ready = []
            for arm,model,deployment in active:
                info=api('GET',model,deployment)
                state['deployments'][arm]['live_status']=info['status']
                ready.append(info['status'] in {'ACTIVE','SCALED_TO_ZERO'})
                if info['status'] in {'FAILED','UNHEALTHY'}:
                    raise RuntimeError(f'{arm} startup failed: '+info['status'])
            save()
            if all(ready): break
            if time.monotonic()-started>20*60: raise TimeoutError('Deployment readiness timeout')
            time.sleep(10)
        state['status']='piloting';save()
        commands={}
        letters=[a for a in ('tev1_native','qwen35_4b_letters') if a in selected]
        if 'qwen35_4b_letters' in selected: letters.append('qwen35_4b_json')
        guards=[a for a in ('safeguard_120b','granite_41_8b','qwen3guard_8b','nemotron_guard') if a in selected]
        if letters: commands['letters_json']=['-m','run.pilot','--arms',*letters,'--limit','2']
        if guards: commands['guards']=['-m','run.guard_pilot','--arms',*guards,'--limit','2']
        for name,args in commands.items():
            log=(out/(name+'.log')).open('w')
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((name,child,log))
        while any(child.poll() is None for _,child,_ in children):
            if time.monotonic()-started>27*60: raise TimeoutError('Stopping pilots before 30-minute shutdown deadline')
            time.sleep(5)
        for name,child,_ in children:state['pilots'][name]={'exit_code':child.returncode,'log':str((out/(name+'.log')).relative_to(ROOT))}
        valid = all(c.returncode==0 for _,c,_ in children)
        for name,child,_ in children:
            text=(out/(name+'.log')).read_text()
            # Reports follow SDK diagnostic lines; locate the last JSON object with counts.
            report=None
            for offset,ch in enumerate(text):
                if ch!='{': continue
                try:
                    candidate=json.JSONDecoder().raw_decode(text[offset:])[0]
                    if isinstance(candidate,dict) and 'counts' in candidate: report=candidate
                except ValueError: pass
            if report is None:valid=False;continue
            failed={k:v for k,v in report['counts'].items() if not k.endswith('|ok')}
            state['pilots'][name]['failed_outputs']=failed
            state['pilots'][name]['experiment_url']=report['experiment_url']
            valid=valid and not failed
        state['status']='pilots_completed' if valid else 'pilots_failed'
    except BaseException as exc:
        state['status']='stopped';state['error']=type(exc).__name__+': '+str(exc)
    finally:
        for _,child,log in children:
            if child.poll() is None:child.kill();child.wait()
            log.close()
        for arm,model,deployment in reversed(active):
            try:state['deployments'][arm]['deactivate']=api('POST',model,deployment,'/deactivate')
            except Exception as exc:state['deployments'][arm]['shutdown_error']=type(exc).__name__
        for arm,model,deployment in active:
            try:
                info=api('GET',model,deployment)
                state['deployments'][arm]['final_status']=info['status']
                state['deployments'][arm]['final_replica_count']=info['active_replica_count']
            except Exception as exc:state['deployments'][arm]['verify_error']=type(exc).__name__
        state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(state,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',choices=list(ALLOWLIST))
    main(parser.parse_args().arms)
