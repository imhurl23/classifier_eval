"""Full source-label coverage for every hosted comparison arm.

Freeze original dataset records and every supported binary request; never trim the
cohort to a budget. Keep source, split, unit, interface and target separate.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from data.bt_auth import ROOT, load_braintrust_key
from data.jsonl import read_jsonl
from run import offline_review
from run.offline_review import digest
from run.experiment_groups import init_experiments, configuration
from run.heldout import atomic, sha
from run.source_labels import source_gold, source_scores, usage_metrics, BENCHMARK, CONTEXT_POLICY
from run.pilot import adapter_for
from run.matrix import matched_option_order
from run.model_metadata import model_metadata
from prompts.runtime import options_for, instructions_for

options_for = lru_cache(None)(options_for)
instructions_for = lru_cache(None)(instructions_for)

ARMS = ('jev_choice','jev_noul','clef_choice','clef_flash_choice','luna_structured','decisions_luna','claude_structured')
RATES = {'jev_choice':(.042,0),'jev_noul':(.042,0),'clef_choice':(.24,0),
         'clef_flash_choice':(.09,0),'luna_structured':(.125,.5),'claude_structured':(2.5,10)}


def cohort_key(row):
    return row['metadata']['source']+'__'+row['input']['unit']


def tasks_for(row):
    source=row['metadata']['source']
    if source.startswith('aegis_v2_'):return ('interaction_harm',)
    if source=='wildguardtest':return ('interaction_harm','prompt_harm','response_harm','response_refusal')
    if source=='xstest':return ('prompt_harm',)
    raise ValueError('Unsupported source: '+source)


def full_jobs(row):
    for arm in ARMS:
        for target in tasks_for(row):
            if arm=='jev_noul' and target!='interaction_harm':continue
            if source_gold(row,target) is None:continue
            rubric='r_long' if target=='interaction_harm' else target+'_s2'
            request=offline_review.request_for(arm,row,target,rubric,'s2')
            job={'arm':arm,'item_id':row['id'],'target':target,'rubric':rubric,'scheme':'s2','replicate':0,
                 'cohort':cohort_key(row),'request':request,'request_sha256':digest(request)}
            job['job_id']=digest({k:job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})
            yield job


def call_reserve(job):
    # Decisions remains explicitly a contingency rather than a published rate.
    input_rate,output_rate=RATES.get(job['arm'],(1,0))
    request=job['request']
    return 3*(len(json.dumps(request).encode())*input_rate+
              request.get('max_tokens',request.get('max_output_tokens',0))*output_rate)/1e6


def prepare(budget_usd=1000):
    authorization=json.loads((ROOT/'study_revision.json').read_text())
    if budget_usd>authorization.get('additional_authorization',{}).get('amount_usd',0) or budget_usd<=0:
        raise ValueError('Full run must fit the additional authorized allocation')
    offline_review.options_for=lru_cache(None)(offline_review.options_for)
    offline_review.instructions_for=lru_cache(None)(offline_review.instructions_for)
    rows=[row for name in ('dis-dev','dis-test') for row in read_jsonl(ROOT/f'data/processed/{name}.jsonl')]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Dataset record IDs must be unique')
    jobs=[job for row in rows for job in full_jobs(row)]
    reserve=sum(call_reserve(j) for j in jobs)
    # Leave room for Decisions uncertainty, infrastructure and gateway fees.
    overhead=250
    if reserve+overhead>budget_usd:raise ValueError(f'Full coverage reserve ${reserve:.2f} plus overhead exceeds allocation')
    paths=[p for folder in ('adapters','prompts') for p in sorted((ROOT/folder).rglob('*')) if p.suffix in {'.py','.md','.yaml'}]
    paths += [ROOT/p for p in ('data/jsonl.py','run/source_full.py','run/source_labels.py','run/model_metadata.py','run/experiment_groups.py',
                              'run/offline_review.py','run/matrix.py','run/pilot.py','scorers/scorers.py')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    version=digest({'files':hashes,'rows':rows,'jobs':jobs,'budget':budget_usd})[:12]
    out=ROOT/'data/processed/source-full'/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists():raise ValueError('Never overwrite an executed plan')
    for name,records in [('sample.jsonl',rows),('manifest.jsonl',jobs)]:
        with (out/name).open('w') as handle:
            for record in records:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    missing=[{'item_id':r['id'],'target':t,'source':r['metadata']['source']} for r in rows for t in tasks_for(r) if source_gold(r,t) is None]
    summary={'version':version,'status':'prepared','items':len(rows),'physical_requests':len(jobs),
             'by_cohort':dict(Counter(cohort_key(r) for r in rows)),
             'requests_by_arm':dict(Counter(j['arm'] for j in jobs)),
             'configuration_counts':dict(Counter(j['cohort']+'|'+j['arm']+'|'+j['target'] for j in jobs)),
             'missing_source_labels':missing,'additional_budget_usd':budget_usd,'request_reserve_usd':reserve,
             'shared_contingency_usd':overhead,'rates_checked':'2026-10-09','billing_reconciled':False,
             'reserve_basis':'Three attempts; UTF-8 wire bytes as conservative token proxy, public hosted rates with cache-write buffer. Decisions $1/M input is an explicit scenario, not a published rate; shared $250 contingency remains.',
             'rate_sources':['https://typesafe.ai/blog/introducing-system-one-models-and-jev',
                             'https://developers.cloudflare.com/workers-ai/models/clef/',
                             'https://developers.cloudflare.com/workers-ai/models/clef-flash/',
                             'https://developers.openai.com/api/docs/pricing',
                             'https://platform.claude.com/docs/en/models/sonnet-5-5/overview'],
             'scope':'Full local dis-dev/dis-test supported binary reference tasks, all seven hosted interfaces. Pair and prompt-only Aegis separate; XSTest separate controls. No retrospective held-out/preregistration claim; prior development exposure retained explicitly.',
             'other_arms':'Baseten and local PolicyLM require separate serving/context readiness and are tracked separately; robustness rubric/granularity grid is not this binary reference run.',
             'file_sha256':hashes,'artifact_sha256':{name:sha(out/name) for name in ('sample.jsonl','manifest.jsonl')}}
    atomic(out/'summary.json',summary)
    print(json.dumps({'plan':str(out),**{k:summary[k] for k in ('items','physical_requests','by_cohort','request_reserve_usd','missing_source_labels')}}),flush=True)
    return out


def load_plan(directory):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    for name,value in summary['file_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen code changed: '+name)
    for name,value in summary['artifact_sha256'].items():
        if sha(directory/name)!=value:raise ValueError('Frozen artifact changed: '+name)
    rows={r['id']:r for r in read_jsonl(directory/'sample.jsonl')};jobs=read_jsonl(directory/'manifest.jsonl')
    if len(rows)!=summary['items'] or len(jobs)!=summary['physical_requests'] or len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Coverage changed')
    expected={j['job_id']:j for row in rows.values() for j in full_jobs(row)}
    if len(expected)!=len(jobs) or any(expected.get(j['job_id'])!=j for j in jobs):raise ValueError('Full source/request coverage changed')
    if sum(call_reserve(j) for j in jobs)+summary['shared_contingency_usd']>summary['additional_budget_usd']:raise ValueError('Budget exceeded')
    return summary,rows,jobs


def run(directory, resume=False, workers_per_arm=4, wall_minutes=720):
    directory=Path(directory)
    with (directory/'.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(directory,resume,workers_per_arm,wall_minutes)


def _run(directory,resume,workers_per_arm,wall_minutes):
    if not 1<=workers_per_arm<=8 or not 0<wall_minutes<=720:raise ValueError('Invalid worker/wall-time bounds')
    summary,rows,jobs=load_plan(directory)
    statepath=directory/'execution.json';resultpath=directory/'results.jsonl'
    previous=json.loads(statepath.read_text()) if statepath.exists() else None
    results=read_jsonl(resultpath) if resultpath.exists() else [];done={r['job_id'] for r in results}
    if (previous or done) and not resume:raise ValueError('Run already started; no automatic paid repeats')
    if resume and (not previous or set(previous.get('inflight',{}))-done):raise ValueError('Ambiguous inflight calls require reconciliation')
    if not done <= {j['job_id'] for j in jobs} or len(done)!=len(results):raise ValueError('Checkpoint mismatch')
    prior_seconds=(previous or {}).get('elapsed_seconds',0)
    if prior_seconds>=wall_minutes*60:raise ValueError('Cumulative wall-time exhausted')
    import braintrust as bt
    load_braintrust_key()
    datasets={};groups={}
    name=(previous or {}).get('run_group') or 'source_full__'+summary['version']+'__'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for cohort in sorted({cohort_key(r) for r in rows.values()}):
        dataset=bt.init_dataset(project='decision-interface-safety',name='source_full__'+summary['version']+'__'+cohort)
        if not previous:
            for row in rows.values():
                if cohort_key(row)==cohort:dataset.insert(id=row['id'],input=row['input'],expected=row['expected'],metadata=row['metadata'])
            dataset.flush()
        datasets[cohort]=dataset
        source=next(r['metadata']['source'] for r in rows.values() if cohort_key(r)==cohort)
        groups[cohort]=init_experiments(project='decision-interface-safety',experiment=name+'__'+cohort,update=bool(previous),dataset=dataset,
              metadata={'study_status':'running','benchmark_id':BENCHMARK,'benchmark_primary':True,'source_dataset':source,
                        'input_unit':cohort.rsplit('__',1)[1],'cohort':cohort,'context_policy':CONTEXT_POLICY,
                        'plan_version':summary['version'],'evaluation_role':'full_source_reference','preregistered':False,
                        'human_gold_available':True,'manual_annotation_required':False,'coverage_complete':False})
    state={'status':'running','run_group':name,'planned_requests':len(jobs),'completed_requests':len(done),'inflight':{},
           'arm_errors':{},'counts':dict(Counter(r['arm']+'|'+r['status'] for r in results)),
           'additional_budget_usd':summary['additional_budget_usd'],'request_reserve_usd':summary['request_reserve_usd'],
           'experiment_urls':dict((previous or {}).get('experiment_urls',{})),'elapsed_seconds':prior_seconds}
    mutex=threading.Lock();started=time.monotonic();deadline=started+wall_minutes*60-prior_seconds
    stopped={arm:threading.Event() for arm in ARMS};failure_counts=Counter()
    def save():
        state['elapsed_seconds']=prior_seconds+time.monotonic()-started
        atomic(statepath,state)
    save()
    local=threading.local()
    def execute(job):
        arm=job['arm']
        if stopped[arm].is_set() or time.monotonic()>=deadline:return
        if not hasattr(local,'adapters'):local.adapters={}
        _,definitions=options_for(job['scheme'],job['rubric']);key=(arm,job['rubric'])
        if key not in local.adapters:
            local.adapters[key]=adapter_for(arm,definitions)
            if hasattr(local.adapters[key],'client'):local.adapters[key].client=local.adapters[key].client.with_options(max_retries=2)
        adapter=local.adapters[key];row=rows[job['item_id']];group=groups[job['cohort']]
        with mutex:
            state['inflight'][job['job_id']]={'arm':arm};save()
        labels,_=options_for(job['scheme'],job['rubric'])
        decision=adapter.classify(row['input']['text'],job['scheme'],instructions_for(job['rubric']),matched_option_order(row['id'],job['target'],job['scheme'],labels))
        result={**{k:job[k] for k in ('job_id','arm','item_id','target','rubric','scheme','request_sha256','cohort')},
                **asdict(decision),'model_version':adapter.model_version,'source_gold':source_gold(row,job['target']),
                'scores':source_scores(asdict(decision),row,job['target'],job['scheme'])}
        with mutex:
            with resultpath.open('a') as handle:
                handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            state['inflight'].pop(job['job_id']);state['completed_requests']+=1
            state['counts'][arm+'|'+decision.status]=state['counts'].get(arm+'|'+decision.status,0)+1
            failure_counts[arm]=failure_counts[arm]+1 if decision.status in {'api_error','invalid_output','truncated'} else 0
            if failure_counts[arm]>=3:
                stopped[arm].set();state['arm_errors'][arm]='Three consecutive failed outputs'
            save()
        group.log(id=job['job_id'],input=job['request'],output={k:result[k] for k in ('label','probs','status')},
                  expected={'label':result['source_gold']},dataset_record_id=row['id'],scores=result['scores'],
                  metadata={**model_metadata(arm),**{k:job[k] for k in ('target','scheme','rubric','job_id','request_sha256')},
                            'source_item_id':row['id'],'model_version':adapter.model_version,'source_dataset':row['metadata']['source'],
                            'scoring_basis':'published_source_labels','source_label_provenance':row['metadata']['gold_provenance'],
                            'source_label_scope':row['metadata'].get('source_label_scope'),'usage':decision.usage},
                  metrics={'latency_ms':decision.latency_ms,'start':time.time()-decision.latency_ms/1000,'end':time.time(),**usage_metrics(decision.usage)})
        with mutex:state['experiment_urls'].update(group.experiment_urls)
    def arm_worker(arm):
        pending=[j for j in jobs if j['arm']==arm and j['job_id'] not in done]
        # map feeds only bounded workers; no provider call begins after stop.
        with ThreadPoolExecutor(max_workers=workers_per_arm) as pool:
            for result in pool.map(execute,pending):pass
    try:
        with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
            futures={a:pool.submit(arm_worker,a) for a in ARMS}
            while not all(f.done() for f in futures.values()):
                time.sleep(15)
                with mutex:save();print(json.dumps({'completed':state['completed_requests'],'planned':len(jobs),'counts':state['counts'],'arm_errors':state['arm_errors']}),flush=True)
            for arm,future in futures.items():
                try:future.result()
                except Exception as exc:state['arm_errors'][arm]=type(exc).__name__+': '+str(exc)
    finally:
        for group in groups.values():group.flush()
        state['status']='completed' if state['completed_requests']==len(jobs) and not state['inflight'] else 'partial'
        state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    # No rows disappear during a logging outage: deterministic IDs permit replay.
    replay(directory,summary,rows,jobs,state,groups)
    print(json.dumps({'status':state['status'],'completed':state['completed_requests'],'planned':len(jobs),'arm_errors':state['arm_errors']}),flush=True)
    return state


def replay(directory,summary,rows,jobs,state,groups):
    """Verify local scoring, repair logs without inference, and mark full cohorts."""
    results=read_jsonl(directory/'results.jsonl') if (directory/'results.jsonl').exists() else []
    by_job={j['job_id']:j for j in jobs};coverage=Counter()
    for result in results:
        job=by_job[result['job_id']];row=rows[result['item_id']]
        if result['request_sha256']!=job['request_sha256'] or result['scores']!=source_scores(result,row,result['target'],result['scheme']):raise ValueError('Score/request mismatch')
        coverage[(job['cohort'],configuration(job))]+=1
        # Write exactly the durable result, never repeat the provider call.
        groups[job['cohort']].log(id=job['job_id'],input=job['request'],output={k:result[k] for k in ('label','probs','status')},
          expected={'label':source_gold(row,job['target'])},dataset_record_id=row['id'],scores=result['scores'],
          metadata={**model_metadata(job['arm']),**{k:job[k] for k in ('target','scheme','rubric','job_id','request_sha256')},
                    'source_item_id':row['id'],'model_version':result['model_version'],'scoring_basis':'published_source_labels',
                    'source_dataset':row['metadata']['source'],'source_label_provenance':row['metadata']['gold_provenance'],'usage':result.get('usage')},
          metrics={'latency_ms':result['latency_ms'],**usage_metrics(result.get('usage'))})
    expected=Counter((j['cohort'],configuration(j)) for j in jobs)
    for group in groups.values():group.flush()
    from run.maintain_experiments import api
    report=[]
    for (cohort,config),count in expected.items():
        actual=coverage[(cohort,config)];child=groups[cohort].for_configuration(config)
        complete=actual==count
        api('PATCH','experiment/'+child.id,json={'metadata':{**groups[cohort].metadata,**config.metadata(),'study_status':'current' if complete else 'partial',
            'coverage_complete':complete,'planned_examples':count,'completed_examples':actual}})
        report.append({'cohort':cohort,'arm':config.arm,'target':config.target,'planned':count,'completed':actual,'complete':complete})
    atomic(directory/'coverage.json',{'configurations':report,'complete':all(r['complete'] for r in report)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--budget-usd',type=float,default=1000)
    parser.add_argument('--plan',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers-per-arm',type=int,default=4);parser.add_argument('--wall-minutes',type=int,default=720)
    args=parser.parse_args()
    if args.prepare:prepare(args.budget_usd)
    elif args.plan:run(args.plan,args.resume,args.workers_per_arm,args.wall_minutes)
    else:parser.error('Choose --prepare or --plan')
