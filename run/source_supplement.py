"""Full paired-source Baseten comparisons and local PolicyLM prompt controls.

GPU deployments are allowlisted, bounded to eight hours including startup, and
always shut down. Native policies/context restrictions are recorded separately.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import json
import os
import signal
from pathlib import Path
import threading
import time
import requests

from data.bt_auth import ROOT,load_braintrust_key
from data.jsonl import read_jsonl
from run import offline_review
from run.offline_review import digest
from run.source_full import cohort_key,tasks_for
from run.source_labels import BENCHMARK,CONTEXT_POLICY,source_gold,source_scores,usage_metrics
from run.heldout import atomic,sha,execute,derive_native,shutdown_baseten,verify_baseten_binding
from run.baseten_preflight import ALLOWLIST
from run.pilot import load_local_service_keys
from run.experiment_groups import init_experiments,configuration
from run.model_metadata import model_metadata
from run.source_native import native_results
from adapters.policylm import make_prompt_request
from prompts.runtime import options_for,instructions_for

COMMON=('tev1_native','qwen35_4b_letters','qwen35_4b_json','safeguard_120b')
NATIVE=('granite_41_8b','qwen3guard_8b','nemotron_guard')
POLICY='policylm_17b_prompt'


def supplement_jobs(row):
    paired=row['input']['unit']=='conversation'
    arms=COMMON+NATIVE if paired else COMMON
    if row['metadata']['source'] in {'wildguardtest','xstest'}:arms+= (POLICY,)
    for arm in arms:
        if arm in COMMON:tasks=[t for t in tasks_for(row) if source_gold(row,t) is not None]
        elif arm==POLICY:tasks=['prompt_harm']
        elif arm=='nemotron_guard':tasks=['native_pair']
        else:tasks=['prompt_harm','response_harm']
        for target in tasks:
            scheme='s3' if arm=='qwen3guard_8b' else 's2'
            rubric='r_long' if target in {'interaction_harm','native_pair'} else target+'_s2'
            if arm==POLICY:
                _,definitions=options_for('s2',rubric)
                request=make_prompt_request(row['input']['prompt'],instructions_for(rubric),definitions)
            else:request=offline_review.request_for(arm,row,target,rubric,'s2' if arm=='qwen3guard_8b' else scheme)
            job={'arm':arm,'item_id':row['id'],'target':target,'scheme':scheme,'rubric':rubric,'replicate':0,
                 'cohort':cohort_key(row),'request':request,'request_sha256':digest(request)}
            job['job_id']=digest({k:job[k] for k in ('arm','item_id','target','scheme','rubric','replicate')})
            yield job


def prepare():
    offline_review.options_for=lru_cache(None)(offline_review.options_for)
    offline_review.instructions_for=lru_cache(None)(offline_review.instructions_for)
    rows=[r for name in ('dis-dev','dis-test') for r in read_jsonl(ROOT/f'data/processed/{name}.jsonl')]
    jobs=[j for r in rows for j in supplement_jobs(r)]
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(ROOT/'data/raw/qwen-tokenizer/tokenizer.json'))
    overflows=[];max_context={}
    for job in jobs:
        if job['arm']==POLICY:
            if len(job['request']['message'])>32000:overflows.append(job['job_id'])
            continue
        request=job['request'];wire=json.dumps(request,ensure_ascii=False)
        tokens=len(tokenizer.encode(wire,add_special_tokens=False).ids)+request.get('max_tokens',0)+256
        limit=8192
        max_context[job['arm']]=max(max_context.get(job['arm'],0),tokens)
        if tokens>limit:overflows.append(job['job_id'])
    if overflows:raise ValueError('Full context readiness failed; no truncation allowed: '+str(len(overflows)))
    files=[p for folder in ('adapters','prompts','deploy/baseten') for p in sorted((ROOT/folder).rglob('*')) if p.suffix in {'.py','.md','.yaml'}]
    files += [ROOT/p for p in ('data/jsonl.py','run/source_supplement.py','run/source_full.py','run/source_labels.py',
       'run/heldout.py','run/source_native.py','run/offline_review.py','run/matrix.py','run/pilot.py','run/baseten_preflight.py','run/experiment_groups.py','run/model_metadata.py','scorers/scorers.py')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    version=digest({'rows':rows,'jobs':jobs,'files':hashes})[:12]
    out=ROOT/'data/processed/source-supplement'/version;out.mkdir(parents=True,exist_ok=True)
    for name,records in [('sample.jsonl',rows),('manifest.jsonl',jobs)]:
        with (out/name).open('w') as handle:
            for record in records:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    summary={'version':version,'items':len(rows),'physical_requests':len(jobs),'requests_by_arm':dict(Counter(j['arm'] for j in jobs)),
      'max_context_proxy':max_context,'context_basis':'Pinned Qwen tokenizer on full JSON request plus maximum output and 256-token template allowance; serving configs retain 8192-token limits; longer guard deployments explicitly updated before launch.',
      'compute_reserve_usd':150,'max_gpu_minutes':480,'max_policy_minutes':720,
      'compute_basis':'7 L4 + 1 H100 GPUs, up to 480 minutes, 1.5x public minute-rate buffer; within the hosted full-run shared $250 contingency, not another $150 allocation.',
      'policy_caveat':'PolicyLM User-only supplement; model-card development/training overlap caveat retained. Native guards separate from common-policy comparisons. No native Aegis turn gold; only derived interaction OR scored there.',
      'file_sha256':hashes,'artifact_sha256':{name:sha(out/name) for name in ('sample.jsonl','manifest.jsonl')}}
    atomic(out/'summary.json',summary)
    print(json.dumps({'plan':str(out),**{k:summary[k] for k in ('physical_requests','requests_by_arm','max_context_proxy')}}),flush=True)
    return out


def load_plan(directory):
    summary=json.loads((directory/'summary.json').read_text())
    for name,value in summary['file_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen code changed: '+name)
    for name,value in summary['artifact_sha256'].items():
        if sha(directory/name)!=value:raise ValueError('Frozen artifact changed: '+name)
    rows={r['id']:r for r in read_jsonl(directory/'sample.jsonl')};jobs=read_jsonl(directory/'manifest.jsonl')
    if len(rows)!=summary['items'] or len(jobs)!=summary['physical_requests']:raise ValueError('Coverage changed')
    offline_review.options_for=lru_cache(None)(offline_review.options_for)
    offline_review.instructions_for=lru_cache(None)(offline_review.instructions_for)
    expected={j['job_id']:j for row in rows.values() for j in supplement_jobs(row)}
    if len(expected)!=len(jobs) or any(expected.get(j['job_id'])!=j for j in jobs):raise ValueError('Request identity changed')
    return summary,rows,jobs


def deployment_api(method,arm,suffix=''):
    model,deployment=ALLOWLIST[arm]
    r=requests.request(method,f'https://api.baseten.co/v1/models/{model}/deployments/{deployment}{suffix}',
       headers={'Authorization':'Bearer '+os.environ['BASETEN_API_KEY']},timeout=30)
    r.raise_for_status();return r.json()


def run(directory,family='baseten',resume=False,workers_per_arm=4):
    directory=Path(directory)
    with (directory/('.'+family+'.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        try:return _run(directory,family,resume,workers_per_arm)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)


def _run(directory,family,resume,workers_per_arm):
    summary,rows,all_jobs=load_plan(directory)
    arms=(POLICY,) if family=='policy' else COMMON+NATIVE
    jobs=[j for j in all_jobs if j['arm'] in arms]
    resultpath=directory/(family+'-results.jsonl');statepath=directory/(family+'-execution.json')
    results=read_jsonl(resultpath) if resultpath.exists() else [];done={r['job_id'] for r in results}
    old=json.loads(statepath.read_text()) if statepath.exists() else None
    if (old or done) and not resume:raise ValueError('Already started; no automatic paid repeats')
    if resume and (not old or set(old.get('inflight',{}))-done):raise ValueError('Ambiguous paid calls require reconciliation')
    import braintrust as bt
    load_local_service_keys();load_braintrust_key()
    if family=='baseten':verify_baseten_binding(set(arms))
    name=(old or {}).get('run_group') or 'source_supplement__'+summary['version']+'__'+family+'__'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    groups={};datasets={}
    for cohort in sorted({j['cohort'] for j in jobs}):
        dataset=bt.init_dataset(project='decision-interface-safety',name='source_supplement__'+summary['version']+'__'+cohort)
        if not old:
            for row in rows.values():
                if cohort_key(row)==cohort:dataset.insert(id=row['id'],input=row['input'],expected=row['expected'],metadata=row['metadata'])
            dataset.flush()
        datasets[cohort]=dataset
        groups[cohort]=init_experiments(project='decision-interface-safety',experiment=name+'__'+cohort,dataset=dataset,update=bool(old),
          metadata={'study_status':'running','benchmark_id':BENCHMARK,'context_policy':CONTEXT_POLICY,'cohort':cohort,
          'source_dataset':next(r['metadata']['source'] for r in rows.values() if cohort_key(r)==cohort),
          'input_unit':cohort.rsplit('__',1)[1],'evaluation_role':'supplemental_policy' if family=='policy' else 'source_baseten_reference',
          'policy_mode':'native/BYOC separated by arm','plan_version':summary['version'],'manual_annotation_required':False,
          'training_overlap_caveat':summary['policy_caveat'] if family=='policy' else None})
    state={'status':'running','run_group':name,'planned_requests':len(jobs),'completed_requests':len(done),'inflight':{},'arm_errors':{},
          'counts':dict(Counter(r['arm']+'|'+r['status'] for r in results)),'experiment_urls':{},'elapsed_seconds':(old or {}).get('elapsed_seconds',0)}
    mutex=threading.Lock();started=time.monotonic();prior=state['elapsed_seconds']
    duration=summary['max_policy_minutes' if family=='policy' else 'max_gpu_minutes']*60
    deadline=started+duration-prior
    if deadline<=started:raise ValueError('Cumulative compute time exhausted')
    def save():state['elapsed_seconds']=prior+time.monotonic()-started;atomic(statepath,state)
    def log_result(result):
        row=rows[result['item_id']];group=groups[result['cohort']]
        gold=source_gold(row,result['target']);is_derived=result.get('physical_calls')==0
        group.log(id=result['job_id'],input=result.get('request',{'derived_from':result.get('parent_job_ids',[])}),
          output={k:result.get(k) for k in ('label','probs','status')},expected={'label':gold} if gold is not None else None,
          scores=result['scores'],dataset_record_id=row['id'],metadata={**model_metadata(result['arm']),
          **{k:result[k] for k in ('target','scheme','rubric','job_id')},'source_item_id':row['id'],'source_dataset':row['metadata']['source'],
          'model_version':result.get('model_version'),'scoring_basis':'published_source_labels' if gold is not None else 'no_independent_turn_gold',
          'source_label_provenance':row['metadata']['gold_provenance'],'physical_calls':0 if is_derived else 1,
          'policy_mode':'byoc' if result['arm']=='granite_41_8b' else 'native' if result['arm'] in NATIVE else 'custom_policy_prompt_only' if result['arm']==POLICY else 'common_policy'},
          metrics={'latency_ms':result['latency_ms'],**({} if is_derived else usage_metrics(result.get('usage')))})
    save();activated=[];stopped={a:threading.Event() for a in arms};failures=Counter();local=threading.local()
    def interrupted(*_):
        for event in stopped.values():event.set()
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def work(job):
        arm=job['arm']
        if stopped[arm].is_set() or time.monotonic()>=deadline:return
        if not hasattr(local,'cache'):local.cache={}
        row=rows[job['item_id']]
        with mutex:state['inflight'][job['job_id']]={'arm':arm};save()
        if arm==POLICY:
            from adapters.policylm import PolicyLMPromptAdapter
            if arm not in local.cache:
                _,defs=options_for('s2',job['rubric']);local.cache[arm]=PolicyLMPromptAdapter(defs)
            adapter=local.cache[arm];decision=adapter.classify_prompt(row['input']['prompt'],instructions_for(job['rubric']));version=adapter.model_version;parts=None
        else:decision,version,parts=execute({**job,'scheme':'s2'} if arm=='qwen3guard_8b' else job,row,local.cache)
        result={**{k:job[k] for k in ('job_id','arm','item_id','target','scheme','rubric','cohort','request_sha256','request')},
                **asdict(decision),'model_version':version,'scores':source_scores(asdict(decision),row,job['target'],job['scheme'])}
        if parts:result['batch_results']=parts
        with mutex:
            with resultpath.open('a') as h:h.write(json.dumps(result,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
            state['inflight'].pop(job['job_id']);state['completed_requests']+=1
            key=arm+'|'+decision.status;state['counts'][key]=state['counts'].get(key,0)+1
            failures[arm]=failures[arm]+1 if decision.status in {'api_error','invalid_output','truncated'} else 0
            if failures[arm]>=3:stopped[arm].set();state['arm_errors'][arm]='Three consecutive failed outputs'
            save()
        log_result(result)
    def arm_work(arm):
        pending=[j for j in jobs if j['arm']==arm and j['job_id'] not in done]
        with ThreadPoolExecutor(max_workers=1 if arm==POLICY else workers_per_arm) as pool:
            for _ in pool.map(work,pending):pass
    try:
        if family=='baseten':
            expected={'tev1_native':2,'qwen35_4b_letters':2,'safeguard_120b':1,'granite_41_8b':1,'qwen3guard_8b':1,'nemotron_guard':1}
            for arm in ALLOWLIST:
                info=deployment_api('GET',arm)
                if info['status']!='INACTIVE' or info['active_replica_count']!=0:raise ValueError('Deployment is not exclusively inactive: '+arm)
                settings=info['autoscaling_settings']
                if settings['min_replica']!=0 or settings['max_replica']!=1:raise ValueError('GPU replica bound changed')
                expected_type='H100 - 1 H100 GPU, 80 GiB VRAM' if arm=='safeguard_120b' else 'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB' if arm in {'tev1_native','qwen35_4b_letters'} else 'L4:4x16 - 1 L4 GPU, 24 GiB video memory, 4 vCPU, 16 GiB'
                if info['instance_type_name']!=expected_type:raise ValueError('GPU hardware changed')
            for arm in ALLOWLIST:
                activated.append(arm);deployment_api('POST',arm,'/activate')
            ready_deadline=min(deadline,started+20*60)
            while True:
                statuses={a:deployment_api('GET',a)['status'] for a in activated}
                state['deployment_statuses']=statuses;save()
                if all(s in {'ACTIVE','SCALED_TO_ZERO'} for s in statuses.values()):break
                if any(s in {'FAILED','UNHEALTHY'} for s in statuses.values()) or time.monotonic()>ready_deadline:raise ValueError('GPU readiness failed')
                time.sleep(10)
        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            futures={a:pool.submit(arm_work,a) for a in arms}
            while not all(f.done() for f in futures.values()):
                time.sleep(15)
                with mutex:save();print(json.dumps({'completed':state['completed_requests'],'planned':len(jobs),'counts':state['counts'],'arm_errors':state['arm_errors']}),flush=True)
            for arm,future in futures.items():
                try:future.result()
                except Exception as exc:state['arm_errors'][arm]=type(exc).__name__+': '+str(exc)
    except Exception as exc:state['arm_errors']['infrastructure']=type(exc).__name__+': '+str(exc)
    finally:
        if activated:state['baseten_shutdown']=shutdown_baseten(set(activated))
        state['status']='completed' if state['completed_requests']==len(jobs) and not state['inflight'] else 'partial';save()
    recorded=read_jsonl(resultpath) if resultpath.exists() else []
    for result in recorded:log_result(result)
    derived=native_results(recorded,rows)
    for result in derived:
        log_result(result)
    with (directory/(family+'-derived.jsonl')).open('w') as h:
        for result in derived:h.write(json.dumps(result)+'\n')
    for group in groups.values():group.flush()
    from run.maintain_experiments import api
    for cohort,group in groups.items():
        for config,child in group.children.items():
            count=sum(r['cohort']==cohort and configuration(r)==config for r in recorded+derived)
            planned=sum(j['cohort']==cohort and configuration(j)==config for j in jobs) if config.rubric not in {'native_or','native_turn'} else sum(cohort_key(r)==cohort for r in rows.values())
            supported=all(source_gold(rows[r['item_id']],r['target']) is not None for r in recorded+derived if r['cohort']==cohort and configuration(r)==config)
            complete=count==planned
            api('PATCH','experiment/'+child.id,json={'metadata':{**group.metadata,**config.metadata(),
               'study_status':'current' if complete and supported and family!='policy' else 'supplemental' if complete else 'partial',
               'policy_mode':'byoc' if config.arm=='granite_41_8b' else 'native' if config.arm in NATIVE else 'custom_policy_prompt_only' if config.arm==POLICY else 'common_policy',
               'coverage_complete':complete,'human_gold_available':supported,'planned_examples':planned,'completed_examples':count}})
            state['experiment_urls'].update(group.experiment_urls)
    state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--plan',type=Path)
    parser.add_argument('--family',choices=['baseten','policy'],default='baseten');parser.add_argument('--resume',action='store_true');parser.add_argument('--workers-per-arm',type=int,default=4)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.plan:run(args.plan,args.family,args.resume,args.workers_per_arm)
    else:parser.error('Choose --prepare or --plan')
