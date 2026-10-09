"""Audit durable source runs and Braintrust counts without model inference."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from data.jsonl import read_jsonl
from run.experiment_groups import configuration
from run.source_labels import source_gold,source_scores
from run.split_experiments import list_experiments
from run.maintain_experiments import api


def verify(directory,family='hosted',remote=False):
    summary=json.loads((directory/'summary.json').read_text())
    rows={r['id']:r for r in read_jsonl(directory/'sample.jsonl')}
    jobs={j['job_id']:j for j in read_jsonl(directory/'manifest.jsonl')}
    name='results.jsonl' if family=='hosted' else family+'-results.jsonl'
    results=read_jsonl(directory/name)
    assert len({r['job_id'] for r in results})==len(results)
    counts=Counter();scores=defaultdict(list);statuses=Counter()
    for result in results:
        job=jobs[result['job_id']];row=rows[result['item_id']]
        assert all(result[k]==job[k] for k in ('arm','item_id','target','scheme','rubric','cohort','request_sha256'))
        assert result['scores']==source_scores(result,row,result['target'],result['scheme'])
        gold=source_gold(row,result['target'])
        if gold is not None:
            # Independent binary contract: caution, refusal and malformed output
            # receive zero; never use an average of unrelated score dimensions.
            strict=float(result['status']=='ok' and result['label']==gold)
            assert result['scores']['source_accuracy']==strict
        key=(result['cohort'],configuration(result))
        counts[key]+=1;statuses[result['arm']+'|'+result['status']]+=1
        if gold is not None:scores[key].append(result['scores']['source_accuracy'])
    report={'plan_version':summary['version'],'family':family,'physical_predictions':len(results),
            'statuses':dict(statuses),'strict_source_scores_verified':True,'remote':None}
    if remote:
        experiments=[e for e in list_experiments() if (e.get('metadata') or {}).get('plan_version')==summary['version'] and
                     (e['metadata'].get('evaluation_role')=='full_source_reference' if family=='hosted' else
                      e['metadata'].get('evaluation_role')==('supplemental_policy' if family=='policy' else 'source_baseten_reference'))]
        source=','.join("'"+e['id']+"'" for e in experiments)
        q='SELECT experiment_id, count(*) AS n, count(scores.source_accuracy) AS scored, avg(scores.source_accuracy) AS accuracy FROM experiment('+source+') GROUP BY experiment_id'
        data=api('POST','btql',json={'query':q,'fmt':'json'})['data']
        totals={r['experiment_id']:r for r in data}
        checked=0
        for exp in experiments:
            meta=exp['metadata'];key=(meta['cohort'],configuration(meta))
            if meta['rubric'] in {'native_turn','native_or'}:continue
            actual=totals.get(exp['id'],{'n':0,'scored':0,'accuracy':None})
            assert actual['n']==counts[key],(exp['name'],actual['n'],counts[key])
            assert actual['scored']==len(scores[key])
            if scores[key]:assert abs(actual['accuracy']-sum(scores[key])/len(scores[key]))<1e-12
            assert exp.get('dataset_id') and meta.get('model') and len(meta.get('arms',[]))==1
            checked+=1
        report['remote']={'physical_configurations_verified':checked,'counts_and_accuracy_match':True}
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--family',choices=('hosted','policy','baseten'),default='hosted')
    parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    report=verify(args.directory,args.family,args.remote)
    (args.directory/(args.family+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
