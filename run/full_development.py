"""Bounded full-conversation development execution with durable restart protection."""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments
import argparse,fcntl,json,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from run.full_context import request,contract
from run.offline_review import digest
from run.heldout import ROOT,sha,atomic
from run.pilot import adapter_for
BASELINE_PRIOR_RESERVE=886.03392


def load(directory):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    if summary.get('context_policy')!='original_full_conversation_v2':raise ValueError('Old pair-only plans are superseded')
    if sha(directory/'sample.jsonl')!=summary['sample_sha256'] or sha(directory/'manifest.jsonl')!=summary['manifest_sha256']:raise ValueError('Plan artifacts changed')
    for name,value in summary['file_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Plan code/policy changed: '+name)
    rows={r['id']:r for r in map(json.loads,(directory/'sample.jsonl').read_text().splitlines())}
    jobs=list(map(json.loads,(directory/'manifest.jsonl').read_text().splitlines()))
    if len(rows)!=summary['items'] or len(jobs)!=summary['requests']:raise ValueError('Plan counts changed')
    seen=set()
    for job in jobs:
        row=rows[job['item_id']]
        if row['metadata']['source_split']!='train' or row['input'].get('history_complete') is not True:raise ValueError('Only original full development dialogues may execute')
        wire=request(job['arm'],row,job['target'],job['rubric'],job['scheme'])
        if wire!=job['request'] or digest(wire)!=job['request_sha256']:raise ValueError('Wire context changed')
        if job['job_id'] in seen:raise ValueError('Duplicate job')
        seen.add(job['job_id'])
    return summary,rows,jobs


def budget(directory,summary):
    prior=BASELINE_PRIOR_RESERVE
    for receipt in (ROOT/'data/processed/full-context-rerun').glob('*/execution.json'):
        if receipt.parent.resolve()!=directory.resolve():prior+=json.loads(receipt.read_text())['allocated_planning_reserve_usd']
    reserve=summary.get('allocated_planning_reserve_usd',summary['uniform_three_attempt_planning_reserve_usd'])
    if prior+reserve>1000:raise ValueError(f'Conservative total reserve ${prior+reserve:.2f} exceeds the authorized $1,000. Reconcile historical billed spend and endpoint fees before larger reruns.')
    return prior,reserve


def run(directory,wall_minutes=45,resume=False):
    directory=Path(directory)
    if wall_minutes<=0:raise ValueError('Positive wall limit required')
    with (directory/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        summary,rows,jobs=load(directory);prior,reserve=budget(directory,summary)
        statepath=directory/'execution.json';results=directory/'results.jsonl'
        previous=json.loads(statepath.read_text()) if statepath.exists() else None
        recorded=list(map(json.loads,results.read_text().splitlines())) if results.exists() else [];done={r['job_id'] for r in recorded}
        if len(done)!=len(recorded) or not done <= {j['job_id'] for j in jobs}:raise ValueError('Checkpoint IDs changed')
        if (previous or done) and not resume:raise ValueError('Never silently repeat executed paid calls')
        if resume and (not previous or previous['status']!='partial'):raise ValueError('Only partial runs may resume')
        if set((previous or {}).get('inflight',{}))-done:raise ValueError('Ambiguous paid calls require reconciliation before resume')
        import braintrust as bt
        from data.bt_auth import load_braintrust_key
        load_braintrust_key()
        name=previous['experiment'] if previous else summary['stage']+'__'+summary['version']+'__'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        experiment=init_experiments(project='decision-interface-safety',experiment=name,update=bool(previous),
           metadata={**experiment_model_metadata(j['arm'] for j in jobs),'context_policy':summary['context_policy'],'original_full_conversations':True,'study_status':'supplemental','evaluation_role':'full_context_diagnostic','benchmark_primary':False,'plan_version':summary['version'],
                     'human_gold_available':False,'pair_only_results_superseded':True,'allocated_planning_reserve_usd':reserve})
        experiment.experiment_urls.update((previous or {}).get('experiment_urls', {}))
        state={'status':'running','experiment':name,'experiment_url':experiment.summarize().experiment_url,'experiment_urls':experiment.experiment_urls,'run_group':name,
          'planned_requests':len(jobs),'completed_requests':len(done),'allocated_planning_reserve_usd':reserve,'prior_planning_reserve_usd':prior,
          'counts':dict(Counter(r['arm']+'|'+r['status'] for r in recorded)),'inflight':{},'arm_errors':{},
          'started_at_utc':(previous or {}).get('started_at_utc',datetime.now(timezone.utc).isoformat()),
          'gold_note':'Validity only until real human full-dialogue labels are retrieved. No old Aegis labels transferred.'}
        mutex=threading.Lock();deadline=time.monotonic()+wall_minutes*60
        def save():atomic(statepath,state)
        save()
        def worker(arm):
            adapters={};fails=0
            for job in [j for j in jobs if j['arm']==arm and j['job_id'] not in done]:
                if time.monotonic()>=deadline:break
                row=rows[job['item_id']];labels,definitions,order,policy=contract(row,job['target'],job['rubric'],job['scheme'])
                key=(job['scheme'],job['rubric'],job['target'])
                if key not in adapters:
                    a=adapter_for(arm,definitions)
                    if hasattr(a,'client'):a.client=a.client.with_options(max_retries=2)
                    adapters[key]=a
                with mutex:state['inflight'][job['job_id']]={'arm':arm};save()
                with experiment.start_span(name=arm+'__'+job['target']+'__'+job['scheme'],type='llm',set_current=False,input=job['request'],
                                           metadata={**model_metadata(arm), **{k:job[k] for k in ('target','scheme','rubric')},'job_id':job['job_id'],'conversation_id':row['id'],'turn_count':len(row['input']['messages'])}) as span:
                    decision=adapters[key].classify(row['input']['text'],job['scheme'],policy,order)
                    result={k:job[k] for k in ('job_id','arm','item_id','target','rubric','scheme','replicate','request_sha256')}
                    result.update(asdict(decision));result.update(model_version=adapters[key].model_version,context_policy='original_full_conversation_v2',
                     turn_count=len(row['input']['messages']),scores={'valid_output':float(decision.status=='ok'),'refused':float(decision.status=='refusal')})
                    with mutex:
                        with results.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                        state['inflight'].pop(job['job_id']);state['completed_requests']+=1
                        c=arm+'|'+decision.status;state['counts'][c]=state['counts'].get(c,0)+1;save()
                    span.log(output={'label':decision.label,'status':decision.status},scores=result['scores'],
                             metadata={**model_metadata(arm),'model_version':result['model_version'],'job_id':job['job_id']},metrics={'latency_ms':decision.latency_ms})
                fails=fails+1 if decision.status in {'api_error','invalid_output','truncated'} else 0
                if fails>=3:
                    with mutex:state['arm_errors'][arm]='Three consecutive failed outputs';save()
                    break
        try:
            arms=sorted({j['arm'] for j in jobs})
            with ThreadPoolExecutor(max_workers=len(arms)) as pool:
                futures={arm:pool.submit(worker,arm) for arm in arms}
                for arm,future in futures.items():
                    try:future.result()
                    except Exception as exc:
                        with mutex:state['arm_errors'][arm]=type(exc).__name__+': '+str(exc);save()
            experiment.flush()
        finally:
            state['status']='completed' if state['completed_requests']==len(jobs) else 'partial';state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
        print(json.dumps(state,indent=2));return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--wall-minutes',type=float,default=45);a=p.parse_args()
    if a.check:
        s,r,j=load(a.plan);print(json.dumps({'items':len(r),'requests':len(j),'status':'validated_without_inference'}))
    else:run(a.plan,a.wall_minutes,a.resume)
