"""Partition an unexecuted full-context plan when provider quota blocks some arms."""
import argparse,json,shutil
from collections import Counter
from pathlib import Path
from run.full_development import load
from run.heldout import ROOT,sha
from run.offline_review import digest


def split(directory,blocked):
    directory=Path(directory).resolve()
    if (directory/'execution.json').exists():
        raise ValueError('Only unexecuted plans may be partitioned; paid results must remain intact')
    summary,rows,jobs=load(directory)
    rates=json.loads((directory/'pricing.json').read_text())
    outputs=[];partitions=[]
    for label,pending in [('available_quota',False),('pending_quota',True)]:
        selected=[j for j in jobs if (j['arm'] in blocked)==pending]
        if not selected:continue
        version=digest({'parent_manifest':summary['manifest_sha256'],'partition':label,'blocked':sorted(blocked)})[:12]
        out=directory.parent/version;out.mkdir(exist_ok=True)
        if (out/'execution.json').exists():raise ValueError('Never rewrite an executed partition')
        used={j['item_id'] for j in selected}
        (out/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for k,r in rows.items() if k in used))
        (out/'manifest.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in selected))
        shutil.copyfile(directory/'pricing.json',out/'pricing.json')
        shutil.copytree(directory/'frozen-code',out/'frozen-code',dirs_exist_ok=True)
        reserve=0;uniform=0
        for j in selected:
            w=j['request'];nbytes=len(json.dumps(w).encode());output=w.get('max_tokens',w.get('max_output_tokens',0))
            r=rates[j['arm']]
            reserve+=3*(nbytes*max(r['input'],r.get('cache_write',0))+output*r['output'])/1e6
            uniform+=3*(nbytes+output)*10/1e6
        new={**summary,'version':version,'stage':summary['stage']+'_'+label,
             'status':'prepared_not_executed','items':len(used),'requests':len(selected),
             'requests_by_arm':dict(Counter(j['arm'] for j in selected)),
             'sample_sha256':sha(out/'sample.jsonl'),'manifest_sha256':sha(out/'manifest.jsonl'),
             'allocated_planning_reserve_usd':reserve,'known_rate_three_attempt_reserve_usd':reserve,
             'uniform_three_attempt_planning_reserve_usd':uniform,
             'parent_plan':str(directory.relative_to(ROOT)),'partition_reason':'Cloudflare daily quota exhausted; blocked arms retained as pending, not removed.'}
        (out/'summary.json').write_text(json.dumps(new,indent=2)+'\n')
        outputs.append({'plan':str(out.relative_to(ROOT)),'partition':label,'requests':len(selected),'reserve_usd':reserve})
        partitions.append({j['job_id'] for j in selected})
    assert set.union(*partitions)=={j['job_id'] for j in jobs}
    if len(partitions)>1:assert partitions[0].isdisjoint(partitions[1])
    (directory/'quota-partitions.json').write_text(json.dumps(outputs,indent=2)+'\n')
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--blocked-arm',action='append',required=True);a=p.parse_args()
    print(json.dumps(split(a.plan,set(a.blocked_arm)),indent=2))
