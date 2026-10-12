"""Reconcile source experiment uploads against saved decisions, without inference."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL

from data.jsonl import read_jsonl
from run.experiment_groups import configuration
from run.heldout import atomic
from run.maintain_experiments import api
from run.model_metadata import model_metadata
from run.source_labels import source_gold,usage_metrics
from run.split_experiments import list_experiments,fetch_many
from run.verify_source_run import verify


def event(result,row):
    derived=result.get('physical_calls')==0
    gold=source_gold(row,result['target'])
    span=str(uuid5(NAMESPACE_URL,result['job_id']))
    return {'id':result['job_id'],'span_id':span,'root_span_id':span,'span_parents':[],
        'input':{'derived_from':result['parent_job_ids']} if derived else result['request'],
        'output':{k:result.get(k) for k in ('label','probs','status')},
        'expected':{'label':gold} if gold is not None else None,'scores':result['scores'],
        'metadata':{**model_metadata(result['arm']),**{k:result[k] for k in ('target','scheme','rubric','job_id')},
            'source_item_id':row['id'],'source_dataset':row['metadata']['source'],
            'model_version':result['model_version'],'physical_calls':0 if derived else 1,
            'policy_mode':'byoc' if result['arm']=='granite_41_8b' else 'native' if result['arm'] in
                {'qwen3guard_8b','nemotron_guard'} else 'common_policy',
            'scoring_basis':'published_source_labels' if gold is not None else 'no_independent_turn_gold',
            'source_label_provenance':row['metadata']['gold_provenance']},
        'metrics':{'latency_ms':result['latency_ms'],**({} if derived else usage_metrics(result.get('usage')))}}


def stale_ids(existing,wanted):
    stale=[]
    for row in existing:
        if row['id'] in wanted:continue
        # Earlier partial native ORs used a different parent set/ID. Archive
        # these invalid derived placeholders, never remove physical predictions.
        if (row.get('metadata') or {}).get('physical_calls')!=0 or (row.get('output') or {}).get('status')!='invalid_output':
            raise ValueError('Unexpected extra row; refusing to remove '+row['id'])
        stale.append(row['id'])
    return stale


def repair(directory):
    directory=Path(directory);verify(directory,'baseten')
    summary=json.loads((directory/'summary.json').read_text())
    rows={r['id']:r for r in read_jsonl(directory/'sample.jsonl')}
    grouped=defaultdict(dict)
    for r in read_jsonl(directory/'baseten-results.jsonl')+read_jsonl(directory/'baseten-derived.jsonl'):
        grouped[(r['cohort'],configuration(r))][r['job_id']]=r
    experiments=[e for e in list_experiments() if (e.get('metadata') or {}).get('plan_version')==summary['version']]
    source=','.join("'"+e['id']+"'" for e in experiments)
    totals={r['experiment_id']:r['n'] for r in api('POST','btql',json={
        'query':'SELECT experiment_id, count(*) AS n FROM experiment('+source+') GROUP BY experiment_id','fmt':'json'})['data']}
    affected=[e for e in experiments if totals.get(e['id'],0)!=len(grouped[(e['metadata']['cohort'],configuration(e['metadata']))])]
    existing=fetch_many([e['id'] for e in affected])
    archive=directory/'upload-repair'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive.mkdir(parents=True)
    atomic(archive/'rows-before.json',existing);atomic(archive/'experiments-before.json',affected)
    report=[]
    for exp in affected:
        wanted=grouped[(exp['metadata']['cohort'],configuration(exp['metadata']))]
        stale=stale_ids(existing.get(exp['id'],[]),wanted)
        known={r['id']:r for r in existing.get(exp['id'],[])}
        events=[]
        for r in wanted.values():
            restored=event(r,rows[r['item_id']])
            for key in ('created','span_id','root_span_id','span_parents'):
                if key in known.get(r['job_id'],{}):restored[key]=known[r['job_id']][key]
            events.append(restored)
        for offset in range(0,len(events),100):
            api('POST','experiment/'+exp['id']+'/insert',json={'events':events[offset:offset+100]})
        if stale:api('POST','experiment/'+exp['id']+'/insert',json={'events':[{'id':i,'_object_delete':True} for i in stale]})
        report.append({'experiment_id':exp['id'],'canonical_rows_uploaded':len(events),'stale_derived_placeholders_removed':len(stale)})
        print(json.dumps(report[-1]),flush=True)
    atomic(archive/'receipt.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    repair(args.directory)
