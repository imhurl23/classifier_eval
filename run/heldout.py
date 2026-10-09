"""Frozen first-wave execution, with durable checkpoints and no automatic ambiguous repeats."""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments
import argparse,fcntl,hashlib,json,os,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import threading
from adapters.base import Decision
from adapters.guards import BasetenGuard,MODELS,granite,qwen3guard,nemotron,safeguard
from data.bt_auth import load_braintrust_key
from prompts.runtime import options_for,instructions_for
from run.offline_review import digest,request_for
from run.matrix import matched_option_order
from run.pilot import adapter_for,load_local_service_keys
from run.baseten_preflight import ALLOWLIST
from scorers.scorers import score_item
from run.source_labels import source_scores
ROOT=Path(__file__).resolve().parents[1]
NATIVE={'granite_41_8b','qwen3guard_8b','nemotron_guard'}
BASE_ARMS={'tev1_native','qwen35_4b_letters','qwen35_4b_json','safeguard_120b'}|NATIVE


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def load_plan(directory):
    summary=json.loads((directory/'summary.json').read_text())
    revision=ROOT/'study_revision.json'
    if revision.exists() and json.loads(revision.read_text()).get('active_context_policy')=='original_full_conversation_v2' and summary.get('context_policy')!='original_full_conversation_v2':
        raise ValueError('Pair-only held-out plan is superseded; rebuild with original full conversations')
    for name in ('sample.jsonl','manifest.jsonl'):
        if sha(directory/name)!=summary['artifact_sha256'][name]:raise ValueError('Frozen artifact changed: '+name)
    for name,expected in summary['file_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen code/policy changed: '+name)
    rows={r['id']:r for r in map(json.loads,(directory/'sample.jsonl').read_text().splitlines())}
    jobs=list(map(json.loads,(directory/'manifest.jsonl').read_text().splitlines()))
    if len(rows)!=summary['items'] or len(jobs)!=summary['physical_requests']:raise ValueError('Frozen counts changed')
    seen=set()
    for j in jobs:
        row=rows[j['item_id']]
        if row['input']['unit']!='conversation' or row['metadata']['split']!='test':raise ValueError('Held-out full test pairs required')
        if j['job_id'] in seen:raise ValueError('Duplicate job')
        seen.add(j['job_id'])
        request=request_for(j['arm'],row,j['target'],j['rubric'],j['scheme'])
        if request!=j['request'] or digest(request)!=j['request_sha256']:raise ValueError('Frozen wire request changed')
        if digest({k:j[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})!=j['job_id']:raise ValueError('Frozen job ID changed')
    return summary,rows,jobs


def verify_launch(directory,summary):
    path=directory/'freeze.json'
    if not path.exists():raise ValueError('Not frozen: audit, context, readiness, and budget must be completed first')
    freeze=json.loads(path.read_text())
    if freeze['summary_sha256']!=sha(directory/'summary.json'):raise ValueError('Freeze does not match plan')
    if set(freeze.get('evidence',{}))!={'audit','context','readiness','budget','analysis'}:raise ValueError('Incomplete freeze evidence')
    for entry in freeze['evidence'].values():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Freeze evidence changed')
    if not freeze.get('policy_finalized') or not freeze.get('audit_complete'):raise ValueError('Policy audit is not finalized')
    if not freeze.get('context_verified') or not freeze.get('readiness_verified'):raise ValueError('Context/readiness incomplete')
    ceiling=freeze.get('total_study_planning_ceiling_usd')
    if not isinstance(ceiling,(float,int)) or not 0<ceiling<=1000 or not freeze.get('billing_verified'):raise ValueError('Cost evidence does not support the authorized $1,000 ceiling')
    from run.freeze_heldout import validate_audit,validate_budget
    validate_audit(json.loads((ROOT/freeze['evidence']['audit']['path']).read_text()))
    amount,wall=validate_budget(json.loads((ROOT/freeze['evidence']['budget']['path']).read_text()),summary)
    if amount!=ceiling or wall!=freeze['max_wall_minutes']:raise ValueError('Frozen budget receipt mismatch')
    return freeze


def execute(job,row,cache):
    arm=job['arm'];labels,definitions=options_for(job['scheme'],job['rubric'])
    order=matched_option_order(row['id'],job['target'],job['scheme'],labels)
    if arm in NATIVE|{'safeguard_120b'}:
        if arm not in cache:cache[arm]=BasetenGuard(arm,deployment_id=ALLOWLIST[arm][1])
        guard=cache[arm]
        prompt,response=row['input']['prompt'],row['input']['response']
        if arm=='granite_41_8b':
            decision=granite(guard,prompt,response,instructions_for(job['rubric'])+'\nUnsafe criterion: '+definitions['unsafe'],job['target'])
        elif arm=='qwen3guard_8b':decision=qwen3guard(guard,prompt,response if job['target']=='response_harm' else None)
        elif arm=='nemotron_guard':
            parts=nemotron(guard,prompt,response)
            first=next(iter(parts.values()))
            status=next((d.status for d in parts.values() if d.status!='ok'),'ok')
            decision=Decision(None,None,first.raw,status,first.latency_ms,first.usage)
            return decision,guard.model_version,{t:asdict(d) for t,d in parts.items()}
        else:decision=safeguard(guard,row['input']['text'],instructions_for(job['rubric']),definitions,order)
        return decision,guard.model_version,None
    key=(arm,job['scheme'],job['rubric'])
    if key not in cache:
        adapter=adapter_for(arm,definitions)
        if hasattr(adapter,'client'):adapter.client=adapter.client.with_options(max_retries=2)
        cache[key]=adapter
    adapter=cache[key]
    return adapter.classify(row['input']['text'],job['scheme'],instructions_for(job['rubric']),order),adapter.model_version,None


def result_scores(job,decision,row):
    return source_scores({'label':decision.label,'probs':decision.probs,'status':decision.status},
                         row,job['target'],job['scheme'])


def shutdown_baseten(arms):
    import requests
    output={}
    for arm in sorted(arms & BASE_ARMS):
        mapped='qwen35_4b_letters' if arm=='qwen35_4b_json' else arm
        if mapped in output:continue
        model,deployment=ALLOWLIST[mapped]
        try:
            headers={'Authorization':'Bearer '+os.environ['BASETEN_API_KEY']}
            url=f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}'
            r=requests.post(url+'/deactivate',headers=headers,timeout=30);r.raise_for_status()
            deadline=time.monotonic()+120
            while True:
                r=requests.get(url,headers=headers,timeout=30);r.raise_for_status();info=r.json()
                if info['status']=='INACTIVE' and info['active_replica_count']==0:break
                if time.monotonic()>=deadline:break
                time.sleep(3)
            output[mapped]={'status':info['status'],'replicas':info['active_replica_count']}
            output[mapped]['verified_inactive']=info['status']=='INACTIVE' and info['active_replica_count']==0
        except Exception as exc:output[mapped]={'error':type(exc).__name__}
    return output


def verify_baseten_binding(arms):
    """Reject environment drift before starting any held-out requests."""
    for arm in arms & BASE_ARMS:
        mapped='qwen35_4b_letters' if arm=='qwen35_4b_json' else arm
        model,deployment=ALLOWLIST[mapped]
        env=MODELS[mapped][0] if mapped in MODELS else 'BASETEN_TEV1_MODEL_ID' if mapped=='tev1_native' else 'BASETEN_QWEN35_MODEL_ID'
        if os.getenv(env)!=model:raise ValueError('Baseten environment model mismatch: '+arm)
        if mapped not in MODELS:
            denv='BASETEN_TEV1_DEPLOYMENT_ID' if mapped=='tev1_native' else 'BASETEN_QWEN35_DEPLOYMENT_ID'
            if os.getenv(denv)!=deployment:raise ValueError('Baseten environment deployment mismatch: '+arm)


def derive_native(results,rows):
    """Derive interaction OR from complete turns, without additional inference/usage."""
    groups={}
    for result in results:
        if result['arm'] in NATIVE:
            groups.setdefault((result['arm'],result['item_id']),{})[result['target']]=result
    output=[]
    for (arm,item_id),group in sorted(groups.items()):
        if arm=='nemotron_guard':
            parent=group.get('native_pair',{})
            parts=parent.get('batch_results',{})
            parents=[parent['job_id']] if parent else []
            latency=parent.get('latency_ms',0)
        else:
            parts={target:group[target] for target in ('prompt_harm','response_harm') if target in group}
            parents=[p['job_id'] for p in parts.values()]
            latency=sum(p.get('latency_ms',0) for p in parts.values())
        valid=set(parts)=={'prompt_harm','response_harm'} and all(p['status']=='ok' and p['label'] in {'safe','unsafe','needs_caution'} for p in parts.values())
        label=('unsafe' if any(p['label']=='unsafe' for p in parts.values()) else 'needs_caution' if any(p['label']=='needs_caution' for p in parts.values()) else 'safe') if valid else None
        decision=Decision(label,None,{'derived_from':parents},'ok' if valid else 'invalid_output',latency)
        scheme='s3' if arm=='qwen3guard_8b' else 's2'
        job={'target':'interaction_harm','scheme':scheme}
        output.append({'arm':arm,'item_id':item_id,'target':'interaction_harm','policy_mode':'native' if arm!='granite_41_8b' else 'byoc',
                       'scheme':scheme,'parent_job_ids':parents,'physical_calls':0,'usage_scope':'already_recorded_on_parents',
                       **asdict(decision),'scores':result_scores(job,decision,rows[item_id])})
    return output


def run(directory,wall_minutes=120,resume=False):
    if wall_minutes<=0:raise ValueError('Positive wall time required')
    directory=Path(directory)
    with (directory/'execution.lock').open('a') as lockfile:
        try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another process owns this plan')
        summary,rows,jobs=load_plan(directory);freeze=verify_launch(directory,summary)
        if wall_minutes>freeze['max_wall_minutes']:raise ValueError('Wall time exceeds frozen compute envelope')
        statepath=directory/'execution.json';resultpath=directory/'results.jsonl'
        previous=json.loads(statepath.read_text()) if statepath.exists() else None
        completed=list(map(json.loads,resultpath.read_text().splitlines())) if resultpath.exists() else []
        done={r['job_id'] for r in completed}
        if len(done)!=len(completed) or not done.issubset({j['job_id'] for j in jobs}):raise ValueError('Invalid checkpoint identity')
        if (previous or completed) and not resume:raise ValueError('Plan already executed; never automatically repeat paid calls')
        if resume and (not previous or previous['status']!='partial'):raise ValueError('Only partial executions may resume')
        unresolved=set((previous or {}).get('inflight',{}))-done
        if unresolved:raise ValueError('Ambiguous paid calls require reconciliation before resume: '+','.join(sorted(unresolved)))
        prior_seconds=(previous or {}).get('elapsed_execution_seconds',0)
        remaining=freeze['max_wall_minutes']*60-prior_seconds
        if remaining<=0:raise ValueError('Frozen cumulative execution time exhausted')
        import braintrust as bt
        load_local_service_keys();load_braintrust_key()
        verify_baseten_binding({j['arm'] for j in jobs})
        experiment_name=previous['experiment'] if previous else 'heldout_firstwave__'+summary['version']+'__'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        experiment=init_experiments(project='decision-interface-safety',experiment=experiment_name,update=bool(previous),
                           metadata={**experiment_model_metadata(j['arm'] for j in jobs),'purpose':'heldout_firstwave','frozen_plan':summary['version'],'freeze_sha256':sha(directory/'freeze.json'),
                                     'policy_mode':'native arms reported separately','single_reviewer_audit':True})
        experiment.experiment_urls.update((previous or {}).get('experiment_urls', {}))
        state={'status':'running','experiment':experiment_name,'experiment_url':experiment.summarize().experiment_url,'experiment_urls':experiment.experiment_urls,'run_group':experiment_name,
               'planned_requests':len(jobs),'completed_requests':len(done),'inflight':{},'arm_errors':{},
               'started_at_utc':(previous or {}).get('started_at_utc',datetime.now(timezone.utc).isoformat()),
               'counts':dict(Counter(r['arm']+'|'+r['status'] for r in completed))}
        mutex=threading.Lock();started=time.monotonic();deadline=started+min(wall_minutes*60,remaining)
        def save():
            state['elapsed_execution_seconds']=prior_seconds+time.monotonic()-started
            atomic(statepath,state)
        save()
        arms={j['arm'] for j in jobs}
        def worker(arm):
            cache={};failures=0
            for job in [j for j in jobs if j['arm']==arm and j['job_id'] not in done]:
                if time.monotonic()>=deadline:
                    with mutex:state['arm_errors'][arm]='Wall-time limit';save()
                    break
                row=rows[job['item_id']]
                with mutex:
                    state['inflight'][job['job_id']]={'arm':arm,'started_at_utc':datetime.now(timezone.utc).isoformat()};save()
                # Once inflight is durable, an interruption cannot silently cause a paid repeat.
                with experiment.start_span(name=arm+'__'+job['target'],type='llm',set_current=False,input=job['request'],metadata={**model_metadata(arm), **{k:job[k] for k in ('target','scheme','rubric')},'job_id':job['job_id']}) as span:
                    decision,version,parts=execute(job,row,cache)
                    output={'job_id':job['job_id'],'arm':arm,'item_id':row['id'],'target':job['target'],'scheme':job['scheme'],
                            'rubric':job['rubric'],'replicate':job['replicate'],'model_version':version,'request_sha256':job['request_sha256'],
                            'source':row['metadata']['source'],**asdict(decision),'scores':result_scores(job,decision,row)}
                    if parts:output.update(batch_results=parts,usage_scope='native_pair_once')
                    with mutex:
                        with resultpath.open('a') as handle:
                            handle.write(json.dumps(output,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
                        state['inflight'].pop(job['job_id']);state['completed_requests']+=1
                        key=arm+'|'+decision.status;state['counts'][key]=state['counts'].get(key,0)+1;save()
                    span.log(output={'label':decision.label,'status':decision.status,'parts':parts},scores=output['scores'],
                             metadata={**model_metadata(arm),'model_version':version,'job_id':job['job_id']},metrics={'latency_ms':decision.latency_ms})
                failures=failures+1 if decision.status in {'api_error','invalid_output','truncated'} else 0
                if failures>=3:
                    with mutex:state['arm_errors'][arm]='Three consecutive failed outputs';save()
                    break
        try:
            with ThreadPoolExecutor(max_workers=len(arms)) as pool:
                futures={arm:pool.submit(worker,arm) for arm in sorted(arms)}
                for arm,future in futures.items():
                    try:future.result()
                    except Exception as exc:
                        with mutex:state['arm_errors'][arm]=type(exc).__name__+': '+str(exc);save()
            experiment.flush()
        finally:
            state['baseten_shutdown']=shutdown_baseten(arms)
            recorded=list(map(json.loads,resultpath.read_text().splitlines())) if resultpath.exists() else []
            atomic(directory/'native-derived.json',derive_native(recorded,rows))
            state['status']='completed' if state['completed_requests']==len(jobs) else 'partial'
            state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--wall-minutes',type=float,default=120)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        summary,rows,jobs=load_plan(args.plan)
        print(json.dumps({'items':len(rows),'physical_requests':len(jobs),'status':'plan_validated_without_inference'}));return
    print(json.dumps(run(args.plan,args.wall_minutes,args.resume),indent=2))
if __name__=='__main__':main()
