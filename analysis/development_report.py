"""Summarize completed development results without claiming held-out performance."""
import json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'data/processed/study-reference/00de7797f0a6'
REPEATS=ROOT/'data/processed/study-repeatability/477ac56d553a'
BATCHING=ROOT/'data/processed/study-batching/4424eb16d43d'

def read(directory):
    path=directory/'results.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

def main():
    reference=read(REFERENCE)
    lookup={(r['arm'],r['item_id'],r['target']):r for r in reference}
    stats=defaultdict(lambda:defaultdict(int))
    for r in reference:
        if r['target']!='interaction_harm': continue
        s=stats[r['arm']];s['n']+=1;s['valid']+=r['status']=='ok'
        gold=r['human_interaction_gold'];pred=r['label'] if r['status']=='ok' else None
        s['correct']+=pred==gold
        if gold=='unsafe':
            s['unsafe_n']+=1;s['unsafe_detected']+=pred=='unsafe'
        elif gold=='safe':
            s['safe_n']+=1;s['false_unsafe']+=pred=='unsafe'
    for s in stats.values():
        s['source_agreement_strict']=s['correct']/s['n']
        s['unsafe_recall_strict']=s['unsafe_detected']/s['unsafe_n']
        s['safe_false_unsafe_rate']=s['false_unsafe']/s['safe_n']
    comparisons={}
    for name,directory in [('repeatability',REPEATS),('batching',BATCHING)]:
        counts=defaultdict(lambda:defaultdict(int))
        for r in read(directory):
            predictions=r.get('batch_results') or {r['target']:r}
            for target,pred in predictions.items():
                base=lookup[(r['arm'],r['item_id'],target)]
                c=counts[r['arm']+'|'+target];c['comparisons']+=1
                if pred['status']=='ok' and base['status']=='ok':
                    c['both_valid']+=1;c['label_changed']+=pred['label']!=base['label']
                else: c['failed_comparison']+=1
        state=json.loads((directory/'execution.json').read_text())
        comparisons[name]={'status':state['status'],'completed_requests':state['completed_requests'],
                            'planned_requests':state['planned_requests'],'experiment_url':state['experiment_url'],
                            'by_arm_target':dict(counts)}
    report={'scope':'Development only. Source-label agreement is not independently audited study-policy accuracy.',
            'reference_interaction':dict(stats),'diagnostics':comparisons,
            'pending':['Baseten development verification','serving context coverage','billing reconciliation','preregistration','held-out matrix','independent human audit','final analysis']}
    destination=ROOT/'data/processed/development-report.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
