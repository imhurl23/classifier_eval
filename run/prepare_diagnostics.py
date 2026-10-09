"""Freeze repeatability/batching requests from the completed reference sample."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from run.development import ROOT, reserve_usd, load_plan
from run.offline_review import digest, jobs, request_for, bundled_request


def prepare():
    parent = ROOT/'data/processed/study-reference/00de7797f0a6'
    source = json.loads((parent/'summary.json').read_text())
    rows = [json.loads(x) for x in (parent/'sample.jsonl').read_text().splitlines()][:40]
    arms = set(source['requests_by_arm'])
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source['file_sha256']}
    prior = source['prior_stage_reserve_usd'] + source['stage_reserve_usd']
    for stage in ('repeatability', 'batching'):
        plan = []
        for kind, arm, row, target, rubric, scheme, rep in jobs(rows):
            if kind != stage or arm not in arms:
                continue
            request = bundled_request(arm,row) if target == 'bundle' else request_for(arm,row,target,rubric,scheme)
            plan.append(dict(arm=arm,item_id=row['id'],target=target,rubric=rubric,scheme=scheme,
                             replicate=rep+1 if stage=='repeatability' else rep,
                             request=request,request_sha256=digest(request)))
        version = digest({'plan':plan,'files':hashes})[:12]
        directory = ROOT/'data/processed'/f'study-{stage}'/version
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'execution.json').exists():
            raise ValueError('This frozen stage has already started')
        sample = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        (directory/'sample.jsonl').write_text(sample)
        (directory/'manifest.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in plan))
        reserve = sum(reserve_usd(j) for j in plan)
        summary = dict(version=version,stage=f'development_{stage}',items=len(rows),requests=len(plan),
                       requests_by_arm=dict(Counter(j['arm'] for j in plan)),file_sha256=hashes,
                       sample_sha256=hashlib.sha256(sample.encode()).hexdigest(),prompt_approval=source['prompt_approval'],
                       prepared_at_utc=datetime.now(timezone.utc).isoformat(),parent_stage=str(parent.relative_to(ROOT)),
                       budget_usd=1000,prior_stage_reserve_usd=prior,stage_reserve_usd=reserve,preregistered=False)
        (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        load_plan(directory)
        print(json.dumps({'plan':str(directory.relative_to(ROOT)),'requests':len(plan),'reserve_usd':reserve,'cumulative_reserve_usd':prior+reserve}))
        prior += reserve

if __name__ == '__main__':
    prepare()
