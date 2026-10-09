"""Bounded, frozen source-label benchmark; human labels are supplied by the dataset.

Prepare with --prepare --budget-usd N; execute the returned plan with --plan PATH.
The budget is an additional conservative planning envelope, not provider billing.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import threading
import time

from data.bt_auth import ROOT, load_braintrust_key
from run.development import reserve_usd
from run.experiment_groups import init_experiments
from run.heldout import atomic
from run.matrix import matched_option_order
from run.model_metadata import model_metadata
from run.offline_review import digest, request_for
from run.pilot import adapter_for
from run.source_labels import BENCHMARK, CONTEXT_POLICY, source_gold, source_scores, usage_metrics
from prompts.runtime import options_for, instructions_for

ARMS = ('jev_choice', 'jev_noul', 'clef_choice', 'clef_flash_choice', 'luna_structured', 'claude_structured')
TASKS = {'interaction_harm':'r_long', 'prompt_harm':'prompt_harm_s2',
         'response_harm':'response_harm_s2', 'response_refusal':'response_refusal_s2'}
SEED = 20261009


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates():
    rows = [json.loads(line) for line in (ROOT/'data/processed/heldout-candidate/55a1f8b24835/sample.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['metadata']['source'] == 'wildguardtest'
            and all(source_gold(r, target) is not None for target in TASKS)]
    # Exclude previous full-dialogue prompt exposure as well as the earlier
    # development exclusions already applied to this source candidate.
    from data.build_datasets import normalize_text
    exposed = set()
    for filename in ('data/processed/full-conversations/92f8046d8b14/development.jsonl',
                     'data/processed/dis-pilot.jsonl'):
        for row in map(json.loads, (ROOT/filename).read_text().splitlines()):
            for text in [row['input'].get('prompt',''), *[m['content'] for m in row['input'].get('messages',[]) if m['role']=='user']]:
                exposed.add(normalize_text(text))
    rows = [r for r in rows if normalize_text(r['input']['prompt']) not in exposed]
    groups = defaultdict(list)
    for row in sorted(rows,key=lambda r:r['id']):
        key = (row['expected']['prompt_s2'],row['expected']['response_s2'],
               row['metadata']['response_refusal_label'],row['metadata']['wildguard_adversarial'],
               row['metadata']['token_len_qwen35'] >= 500)
        groups[key].append(row)
    rng = random.Random(SEED)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop())
    return selected


def jobs_for(row):
    for arm in ARMS:
        for target,rubric in TASKS.items():
            if arm=='jev_noul' and target!='interaction_harm':
                continue
            request = request_for(arm,row,target,rubric,'s2')
            job = dict(arm=arm,item_id=row['id'],target=target,rubric=rubric,scheme='s2',replicate=0,
                       request=request,request_sha256=digest(request))
            job['job_id'] = digest({k:job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})
            yield job


def budget_ledger():
    historical = ROOT/'data/processed/study-batching/4424eb16d43d/summary.json'
    allocated = 0
    if historical.exists():
        recorded = json.loads(historical.read_text())
        allocated += recorded['prior_stage_reserve_usd']+recorded['stage_reserve_usd']
    for path in (ROOT/'data/processed/full-context-rerun').glob('*/execution.json'):
        allocated += json.loads(path.read_text()).get('allocated_planning_reserve_usd',0)
    # Retain the earlier infrastructure preflight allowance conservatively.
    infrastructure = 50 if historical.exists() else 0
    for path in (ROOT/'data/processed/source-labeled-benchmark').glob('*/execution.json'):
        allocated += json.loads(path.read_text()).get('additional_budget_usd',0)
    revision = ROOT/'study_revision.json'
    ceiling = json.loads(revision.read_text()).get('authorized_total_budget_usd',1000) if revision.exists() else 1000
    return {'authorized_ceiling_usd':ceiling,'prior_allocations_usd':allocated,
            'infrastructure_allowance_usd':infrastructure,'billing_reconciled':False}


def prepare(budget_usd, max_items=200):
    if budget_usd <= 0 or max_items <= 0:
        raise ValueError('Positive budget and cohort size required')
    ledger = budget_ledger()
    if ledger['prior_allocations_usd']+ledger['infrastructure_allowance_usd']+budget_usd > ledger['authorized_ceiling_usd']:
        raise ValueError('New allocation exceeds the existing authorized study envelope')
    selected, jobs, reserve = [], [], 0
    for row in candidates()[:max_items]:
        proposed = list(jobs_for(row))
        cost = sum(reserve_usd(j) for j in proposed)
        # Reserve a fifth of the authorization for overhead/unrecorded charges.
        if reserve+cost > budget_usd*.8:
            break
        selected.append(row);jobs.extend(proposed);reserve+=cost
    if len(selected) < 4:
        raise ValueError('Budget cannot cover a meaningful matched cohort')
    files = [p for folder in ('adapters','prompts') for p in sorted((ROOT/folder).rglob('*'))
             if p.suffix in {'.py','.md','.yaml'}]
    files += [ROOT/p for p in ('run/source_benchmark.py','run/source_labels.py','run/model_metadata.py',
                              'run/experiment_groups.py','run/offline_review.py','run/matrix.py',
                              'run/pilot.py','run/development.py','run/heldout.py','scorers/scorers.py')]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    version = digest({'files':hashes,'jobs':jobs,'budget_usd':budget_usd})[:12]
    out = ROOT/'data/processed/source-labeled-benchmark'/version
    out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists():
        raise ValueError('Do not overwrite an executed benchmark plan')
    (out/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    summary = {'version':version,'benchmark_id':BENCHMARK,'context_policy':CONTEXT_POLICY,
               'source':'wildguardtest','items':len(selected),'requests':len(jobs),
               'requests_by_arm':dict(Counter(j['arm'] for j in jobs)),
               'additional_budget_usd':budget_usd,'planning_reserve_usd':reserve,'budget_ledger':ledger,
               'reserve_basis':'Three attempts at a $10/M input/output contingency, UTF-8 request bytes as input proxy; 20% budget headroom. Not actual billed spend.',
               'cohort_policy':'Bounded scoring check, not a reliable model ranking. Deterministic matched sample stratified by source harm/refusal, adversarial status, and length; original prompt-response pairs unchanged. Report sample agreement, not full-population estimates.',
               'gold_policy':'Published WildGuardTest human task labels; interaction label is their derived OR. No manual annotation gate.',
               'scope':'New benchmark sample; existing pilots/development exposure excluded. No retrospective preregistration claim.',
               'unexecuted_arms':{'decisions_luna':'Endpoint-specific pricing unresolved; existing Aegis comparison retained.'},
               'file_sha256':hashes,'artifact_sha256':{name:sha(out/name) for name in ('sample.jsonl','manifest.jsonl')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'plan':str(out),'items':len(selected),'requests':len(jobs),'planning_reserve_usd':reserve}),flush=True)
    return out


def load_plan(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    for name,value in summary['file_sha256'].items():
        if sha(ROOT/name)!=value:
            raise ValueError('Frozen code changed: '+name)
    for name,value in summary['artifact_sha256'].items():
        if sha(directory/name)!=value:
            raise ValueError('Frozen artifact changed: '+name)
    rows = {r['id']:r for r in map(json.loads,(directory/'sample.jsonl').read_text().splitlines())}
    jobs = list(map(json.loads,(directory/'manifest.jsonl').read_text().splitlines()))
    if len(rows)!=summary['items'] or len(jobs)!=summary['requests'] or len({j['job_id'] for j in jobs})!=len(jobs):
        raise ValueError('Frozen coverage changed')
    for job in jobs:
        row = rows[job['item_id']]
        if row['metadata']['source']!='wildguardtest' or source_gold(row,job['target']) is None:
            raise ValueError('Source human label required on each exact input')
        request = request_for(job['arm'],row,job['target'],job['rubric'],job['scheme'])
        if digest({k:job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')}) != job['job_id']:
            raise ValueError('Frozen job identity changed')
        if request != job['request'] or digest(request)!=job['request_sha256']:
            raise ValueError('Frozen request changed')
    if sum(reserve_usd(j) for j in jobs) > summary['additional_budget_usd']*.8:
        raise ValueError('Frozen budget exceeded')
    return summary, rows, jobs


def _run(directory, resume=False, wall_minutes=60):
    if wall_minutes <= 0:
        raise ValueError('Positive wall-time limit required')
    directory = Path(directory)
    summary, rows, jobs = load_plan(directory)
    results_path, state_path = directory/'results.jsonl', directory/'execution.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    results = [json.loads(l) for l in results_path.read_text().splitlines()] if results_path.exists() else []
    done = {r['job_id'] for r in results}
    if (previous or done) and not resume:
        raise ValueError('Benchmark already started; no automatic paid repeats')
    if resume and (not previous or previous.get('inflight')):
        raise ValueError('Ambiguous inflight calls require reconciliation before resuming')
    if not done <= {j['job_id'] for j in jobs}:
        raise ValueError('Checkpoint does not match this plan')
    import braintrust as bt
    load_braintrust_key()
    dataset = bt.init_dataset(project='decision-interface-safety', name='source_labeled_wildguard__'+summary['version'])
    if not previous:
        for row in rows.values():
            dataset.insert(id=row['id'],input=row['input'],expected=row['expected'],metadata=row['metadata'])
        dataset.flush()
    name = previous['run_group'] if previous else 'source_labeled_wildguard__'+summary['version']+'__'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    experiment = init_experiments(project='decision-interface-safety',experiment=name,update=bool(previous),dataset=dataset,
        metadata={'study_status':'current','benchmark_id':BENCHMARK,'benchmark_primary':True,
                  'evaluation_role':'source_labeled_benchmark','evaluation_split':'test_sample','source_dataset':'wildguardtest',
                  'human_gold_available':True,'manual_annotation_required':False,'context_policy':CONTEXT_POLICY,
                  'source_label_scope':'Published human target labels; derived OR for interaction_harm',
                  'additional_budget_usd':summary['additional_budget_usd'],'sample_items':summary['items'],
                  'plan_version':summary['version'],'preregistered':False})
    experiment.experiment_urls.update((previous or {}).get('experiment_urls',{}))
    state = {'status':'running','run_group':name,'planned_requests':len(jobs),'completed_requests':len(done),
             'inflight':{},'arm_errors':{},'experiment_urls':experiment.experiment_urls,
             'additional_budget_usd':summary['additional_budget_usd'],'planning_reserve_usd':summary['planning_reserve_usd']}
    lock = threading.Lock()
    deadline = time.monotonic()+wall_minutes*60
    def save():atomic(state_path,state)
    save()
    def worker(arm):
        adapters, failures = {}, 0
        for job in [j for j in jobs if j['arm']==arm and j['job_id'] not in done]:
            if time.monotonic() >= deadline:break
            row = rows[job['item_id']]
            options,definitions = options_for('s2',job['rubric'])
            key = job['rubric']
            if key not in adapters:
                adapters[key] = adapter_for(arm,definitions)
                if hasattr(adapters[key],'client'):
                    adapters[key].client=adapters[key].client.with_options(max_retries=2)
            with lock:state['inflight'][job['job_id']]={'arm':arm};save()
            with experiment.start_span(id=job['job_id'],name=arm+'__'+job['target'],type='llm',set_current=False,
                    input=job['request'],metadata={**model_metadata(arm),**{k:job[k] for k in ('target','scheme','rubric')},
                                                  'job_id':job['job_id'],'source_item_id':row['id']}) as span:
                decision = adapters[key].classify(row['input']['text'],'s2',instructions_for(job['rubric']),
                                                 matched_option_order(row['id'],job['target'],'s2',options))
                result={**{k:job[k] for k in ('job_id','arm','item_id','target','scheme','rubric','request_sha256')},
                        **asdict(decision),'model_version':adapters[key].model_version,
                        'source_gold':source_gold(row,job['target']),
                        'scores':source_scores(asdict(decision),row,job['target'])}
                with lock:
                    with results_path.open('a') as handle:
                        handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
                    state['inflight'].pop(job['job_id']);state['completed_requests']+=1;save()
                    if state['completed_requests']%25==0:
                        print(json.dumps({'completed':state['completed_requests'],'planned':len(jobs)}),flush=True)
                span.log(output={k:result[k] for k in ('label','probs','status')},expected={'label':result['source_gold']},
                         dataset_record_id=row['id'],scores=result['scores'],
                         metadata={'source_dataset':'wildguardtest','model_version':result['model_version'],
                                   'scoring_basis':'published_source_labels','source_label_provenance':row['metadata']['gold_provenance'],
                                   'source_refusal_label':row['metadata'].get('response_refusal_label'),'usage':decision.usage},
                         metrics={'latency_ms':decision.latency_ms,**usage_metrics(decision.usage)})
            failures=failures+1 if decision.status in {'api_error','invalid_output','truncated'} else 0
            if failures>=3:
                with lock:state['arm_errors'][arm]='Three consecutive failed outputs';save()
                break
    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        futures={arm:pool.submit(worker,arm) for arm in ARMS}
        for arm,future in futures.items():
            try:future.result()
            except Exception as exc:
                with lock:state['arm_errors'][arm]=type(exc).__name__+': '+str(exc);save()
    experiment.flush()
    state['status']='completed' if state['completed_requests']==len(jobs) else 'partial'
    state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(state,indent=2),flush=True)
    return state


def run(directory, resume=False, wall_minutes=60):
    directory = Path(directory)
    with (directory/'.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(directory,resume,wall_minutes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--budget-usd',type=float)
    parser.add_argument('--max-items',type=int,default=200);parser.add_argument('--plan',type=Path)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--wall-minutes',type=float,default=60)
    args=parser.parse_args()
    if args.prepare:prepare(args.budget_usd,args.max_items)
    else:run(args.plan,args.resume,args.wall_minutes)
