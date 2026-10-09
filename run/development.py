"""Execute frozen development stages, with checkpoints and Braintrust logs."""
from __future__ import annotations
from run.model_metadata import model_metadata, experiment_model_metadata
from run.experiment_groups import init_experiments, configuration

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

from data.bt_auth import load_braintrust_key
from prompts.runtime import instructions_for, options_for
from run.matrix import matched_option_order
from run.offline_review import digest, request_for, bundled_request
from run.matrix import COMPANION_RUBRICS
from run.pilot import adapter_for
from run.source_labels import source_scores, source_gold, usage_metrics, CONTEXT_POLICY, BENCHMARK

ROOT = Path(__file__).resolve().parents[1]


def load_plan(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    revision=ROOT/'study_revision.json'
    if revision.exists() and json.loads(revision.read_text()).get('active_context_policy')=='original_full_conversation_v2':
        raise ValueError('Pair-only development is superseded; use run.full_development')
    if summary.get('sample_sha256') and hashlib.sha256((directory / 'sample.jsonl').read_bytes()).hexdigest() != summary['sample_sha256']:
        raise ValueError('Frozen sample changed')
    for name, expected in summary['file_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen file changed: {name}; prepare a new manifest')
    rows = {r['id']: r for r in map(json.loads, (directory / 'sample.jsonl').read_text().splitlines())}
    plan = list(map(json.loads, (directory / 'manifest.jsonl').read_text().splitlines()))
    if len(plan) != summary['requests'] or len(rows) != summary['items']:
        raise ValueError('Manifest/sample counts changed')
    ids = set()
    for job in plan:
        row = rows[job['item_id']]
        if row['metadata']['split'] != 'validation' or row['input']['unit'] != 'conversation':
            raise ValueError('Only development pairs may run here')
        request = bundled_request(job['arm'], row) if job['target'] == 'bundle' else request_for(job['arm'], row, job['target'], job['rubric'], job['scheme'])
        if digest(request) != job['request_sha256'] or request != job['request']:
            raise ValueError('Frozen request changed')
        identifier = digest({k: job[k] for k in ('arm','item_id','target','rubric','scheme','replicate')})
        if identifier in ids:
            raise ValueError('Duplicate manifest job')
        ids.add(identifier)
        job['job_id'] = identifier
    return summary, rows, plan


def reserve_usd(job):
    # Uniform conservative planning reserve: UTF-8 bytes rather than estimated
    # tokens, $10/M input/output, and all three possible attempts. This replaces
    # the arbitrary $1/attempt Decisions contingency used for the first stage.
    # Neither this reserve nor the public model rate establishes endpoint billing.
    wire_bytes = len(json.dumps(job['request']).encode())
    return 3 * (wire_bytes * 10 / 1e6 + job['request'].get('max_tokens', job['request'].get('max_output_tokens', 0)) * 10 / 1e6)


def scores_for(stage, job, decision, row):
    return source_scores({'label':decision.label, 'probs':decision.probs, 'status':decision.status},
                         row, job['target'], job['scheme'])


def run(directory, budget_usd, wall_minutes=60, resume=False):
    if budget_usd <= 0 or wall_minutes <= 0:
        raise ValueError('Budget and wall time must be positive')
    summary, rows, plan = load_plan(directory)
    stage = summary.get('stage', 'development_schema_smoke')
    reserve = sum(reserve_usd(j) for j in plan)
    prior_reserve = summary.get('prior_stage_reserve_usd', 0)
    if reserve + prior_reserve > budget_usd:
        raise ValueError(f'Conservative stage reserve ${reserve:.2f} exceeds budget')
    status_path = directory / 'execution.json'
    results_path = directory / 'results.jsonl'
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    completed = list(map(json.loads, results_path.read_text().splitlines())) if results_path.exists() else []
    completed_ids = {r['job_id'] for r in completed}
    if (previous or completed) and not resume:
        raise ValueError('Execution already exists; do not automatically repeat paid calls')
    if resume and (not previous or previous['status'] != 'partial'):
        raise ValueError('Only a stopped partial execution may resume')
    if not completed_ids.issubset({j['job_id'] for j in plan}):
        raise ValueError('Checkpoint results do not belong to this plan')
    import braintrust as bt
    load_braintrust_key()
    prefix = stage if stage != 'development_schema_smoke' else 'development_schema'
    name = previous['experiment'] if previous else f'{prefix}__{summary["version"]}__{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    label_coverage = {}
    for job in plan:
        config = configuration(job)
        label_coverage[config] = label_coverage.get(config, True) and source_gold(rows[job['item_id']], job['target']) is not None
    profiles = {config: {'human_gold_available': labeled,
                        'study_status': 'current' if labeled and stage == 'development_reference' else 'supplemental',
                        'benchmark_primary': labeled and stage == 'development_reference'}
                for config, labeled in label_coverage.items()}
    experiment = init_experiments(project='decision-interface-safety', experiment=name, update=bool(previous), configuration_metadata=profiles,
                         metadata={**experiment_model_metadata(j['arm'] for j in plan),'purpose':stage, 'preregistered':False,
                                   'context_policy':CONTEXT_POLICY, 'benchmark_id':BENCHMARK,
                                   'study_status':'current' if stage=='development_reference' else 'supplemental',
                                   'evaluation_role':'source_labeled_development' if stage=='development_reference' else 'diagnostic',
                                   'plan_version':summary['version'], 'budget_usd':budget_usd,
                                   'stage_reserve_usd':reserve, 'file_sha256':summary['file_sha256'],
                                   'item_ids':list(rows), 'arms':sorted({j['arm'] for j in plan}),
                                   'prompt_approval':summary.get('prompt_approval'),
                                   'sample_sha256':summary.get('sample_sha256'),
                                   'prior_stage_reserve_usd':prior_reserve})
    experiment.experiment_urls.update((previous or {}).get('experiment_urls', {}))
    state = {'experiment':name, 'experiment_url':experiment.summarize().experiment_url,'experiment_urls':experiment.experiment_urls,'run_group':name,
             'budget_usd':budget_usd, 'stage_reserve_usd':reserve,
             'stage':stage, 'prior_stage_reserve_usd':prior_reserve,
             'billing_note':'Reserves are planning assumptions, not confirmed charges or a provider-enforced dollar limit; Decisions rates and gateway fees require billing reconciliation.',
             'started_at_utc':datetime.now(timezone.utc).isoformat(), 'status':'running',
             'planned_requests':len(plan), 'completed_requests':len(completed),
             'counts':dict(Counter(f'{r["arm"]}|{r["status"]}' for r in completed)), 'arm_errors':{},
             'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if previous:
        state['started_at_utc'] = previous['started_at_utc']
        state['resumed_at_utc'] = datetime.now(timezone.utc).isoformat()
    lock = threading.Lock()
    start = time.monotonic()
    deadline = start + wall_minutes * 60
    def checkpoint():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(status_path)
    checkpoint()
    groups = {arm:[j for j in plan if j['arm']==arm and j['job_id'] not in completed_ids] for arm in sorted({j['arm'] for j in plan})}
    def execute_arm(arm, jobs):
        adapters = {}
        failures = 0
        for job in jobs:
            if time.monotonic() >= deadline:
                with lock:
                    state['arm_errors'][arm] = 'Wall-time limit reached before next request'
                    checkpoint()
                break
            row = rows[job['item_id']]
            labels, definitions = options_for(job['scheme'], 'r_long' if job['target'] == 'bundle' else job['rubric'])
            key = (job['scheme'], job['rubric'])
            if key not in adapters:
                adapter = adapter_for(arm, definitions)
                if arm == 'claude_structured':
                    adapter.client = adapter.client.with_options(max_retries=2)
                adapters[key] = adapter
            adapter = adapters[key]
            order = matched_option_order(row['id'], job['target'], job['scheme'], labels)
            with experiment.start_span(name=f'{arm}__{job["scheme"]}__{job["rubric"]}', type='llm',
                                       set_current=False, input=job['request'], metadata={**model_metadata(arm), **{k:job[k] for k in ('target','scheme','rubric')},'job_id':job['job_id']}) as span:
                batch_results = None
                if job['target'] == 'bundle':
                    questions = {}
                    for target, rubric in COMPANION_RUBRICS.items():
                        offered, criteria = options_for('s2', rubric)
                        ordered = matched_option_order(row['id'], target, 's2', offered)
                        questions[target] = {'instructions': instructions_for(rubric), 'criteria': {k: criteria[k] for k in ordered}}
                    decisions = adapter.classify_many(row['input']['text'], questions)
                    if set(decisions) != set(questions):
                        raise ValueError('Bundle returned unexpected targets')
                    from adapters.base import Decision
                    first = next(iter(decisions.values()))
                    status = next((d.status for d in decisions.values() if d.status != 'ok'), 'ok')
                    decision = Decision(None, None, first.raw, status, first.latency_ms, first.usage)
                    batch_results = {t: {**asdict(d), 'option_order': list(questions[t]['criteria'])} for t,d in decisions.items()}
                else:
                    decision = adapter.classify(row['input']['text'], job['scheme'], instructions_for(job['rubric']), order)
                result = {'job_id':job['job_id'], 'arm':arm, 'item_id':row['id'],
                          'target':job['target'], 'scheme':job['scheme'], 'rubric':job['rubric'], 'replicate':job['replicate'],
                          'option_order':order, 'request_sha256':job['request_sha256'],
                          **model_metadata(arm), 'model_version':adapter.model_version, **asdict(decision)}
                if batch_results is not None:
                    result['batch_results'] = batch_results
                    result['usage_scope'] = 'whole_bundle_once'
                scores = scores_for(stage, job, decision, row)
                result['scores'] = scores
                gold = source_gold(row, job['target'])
                result['human_interaction_gold'] = gold
                # Checkpoint before logging so a logging outage cannot cause paid repeats.
                with lock:
                    with results_path.open('a') as handle:
                        handle.write(json.dumps(result, allow_nan=False)+'\n')
                    state['completed_requests'] += 1
                    count_key = f'{arm}|{decision.status}'
                    state['counts'][count_key] = state['counts'].get(count_key,0)+1
                    checkpoint()
                    if state['completed_requests'] % 25 == 0:
                        print(json.dumps({'completed':state['completed_requests'],'planned':len(plan),'counts':state['counts']}),flush=True)
                span.log(output=batch_results if batch_results is not None else {'label':decision.label,'probs':decision.probs,'status':decision.status},
                         scores=scores,
                         expected={'label':gold} if gold is not None else None,
                         metadata={k:v for k,v in result.items() if k not in {'label','probs','latency_ms'}},
                         metrics={'latency_ms':decision.latency_ms, **usage_metrics(decision.usage)})
            failures = failures + 1 if decision.status in {'api_error','invalid_output','truncated'} else 0
            if failures >= 3:
                with lock:
                    state['arm_errors'][arm] = 'Stopped after three consecutive failed outputs; inspect before repeating'
                    checkpoint()
                break
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {arm:pool.submit(execute_arm,arm,jobs) for arm,jobs in groups.items()}
        for arm, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                with lock:
                    # Exception details may include request context; keep them local.
                    state['arm_errors'][arm] = f'{type(exc).__name__}: {exc}'
                    checkpoint()
    experiment.flush()
    state['status'] = 'completed' if state['completed_requests']==len(plan) else 'partial'
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    state['elapsed_seconds'] = time.monotonic()-start
    checkpoint()
    print(json.dumps(state,indent=2),flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--budget-usd',type=float,required=True)
    parser.add_argument('--wall-minutes',type=float,default=60)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.plan,args.budget_usd,args.wall_minutes,args.resume)


if __name__ == '__main__':
    main()
