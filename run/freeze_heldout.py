"""Validate completed audit and launch evidence, then write an immutable freeze receipt."""
import argparse,json,math
from pathlib import Path
from datetime import datetime,timezone
from run.heldout import ROOT,sha,load_plan,atomic
from run.baseten_preflight import ALLOWLIST


def validate_audit(audit):
    packet=json.loads((ROOT/'data/processed/offline-review/f0c7bbda4b65/annotation-template.json').read_text())
    if audit.get('version')!=packet['version'] or audit.get('policy_hash')!=packet['policy_hash']:
        raise ValueError('Audit does not match the policy packet')
    if not isinstance(audit.get('rater_id'),str) or not audit['rater_id'].strip():raise ValueError('Reviewer identity missing')
    ratings=audit.get('ratings',[])
    if len(ratings)!=200 or {r['audit_id'] for r in ratings}!={r['audit_id'] for r in packet['ratings']}:
        raise ValueError('Exactly 200 unique packet ratings are required')
    for r in ratings:
        for target in ('interaction_harm','prompt_harm','response_harm','response_refusal'):
            labels={'refusal','not_refusal','unrateable'} if target=='response_refusal' else {'safe','unsafe','unrateable'}
            if r.get(target) not in labels:raise ValueError('Incomplete/invalid human rating')
        if (r.get('ambiguity_targets') or any(r[t]=='unrateable' for t in ('interaction_harm','prompt_harm','response_harm','response_refusal'))) and not r.get('reason','').strip():
            raise ValueError('Ambiguous/unrateable judgments require a reason')
    return audit


def validate_budget(budget,summary):
    arms=set(summary['requests_by_arm'])
    if budget.get('manifest_sha256')!=summary['artifact_sha256']['manifest.jsonl'] or budget.get('billing_verified') is not True:
        raise ValueError('Verified billing must cover this exact manifest')
    def amount(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0
    ceiling=budget.get('total_study_planning_ceiling_usd');wall=budget.get('max_wall_minutes')
    if not amount(ceiling) or not 0<ceiling<=1000 or not amount(wall) or wall<=0:
        raise ValueError('Budget/time envelope invalid')
    allocations=budget.get('arms',{})
    if set(allocations)!=arms or any(not e.get('evidence') or not e.get('rate_basis') or not amount(e.get('ceiling_usd')) for e in allocations.values()):
        raise ValueError('Per-arm rates, ceilings or compute/fee evidence missing')
    prior=budget.get('prior_spend_and_unsettled_ceiling_usd');fees=budget.get('shared_fees_and_contingency_usd')
    if not amount(prior) or not amount(fees) or not budget.get('prior_spend_evidence'):
        raise ValueError('Historical spend and unsettled charges must be reconciled')
    total=prior+fees+sum(e['ceiling_usd'] for e in allocations.values())
    if total>ceiling+1e-8:raise ValueError('Allocations exceed total study ceiling')
    return ceiling,wall


def freeze(directory,audit_path,context_path,readiness_path,budget_path,policy_finalized=False):
    directory=Path(directory);summary,rows,jobs=load_plan(directory)
    if (directory/'execution.json').exists() or (directory/'freeze.json').exists():raise ValueError('Do not refreeze an existing execution/freeze')
    audit=validate_audit(json.loads(Path(audit_path).read_text()))
    if not policy_finalized:raise ValueError('Finalize policy after reviewing audit before freezing')
    context=json.loads(Path(context_path).read_text());coverage={r['job_id']:r for r in context.get('jobs',[])}
    if len(coverage)!=len(jobs):raise ValueError('Context verification must cover every physical request')
    for job in jobs:
        c=coverage.get(job['job_id'],{})
        if c.get('request_sha256')!=job['request_sha256'] or c.get('fits_full_input_and_output') is not True or not c.get('evidence'):
            raise ValueError('Missing verified full-context coverage: '+job['job_id'])
    readiness=json.loads(Path(readiness_path).read_text());arms=set(summary['requests_by_arm'])
    for arm in arms:
        r=readiness.get('arms',{}).get(arm,{})
        if not r.get('validated') or not r.get('model_version') or not r.get('evidence'):raise ValueError('Arm lacks validated readiness: '+arm)
        mapped='qwen35_4b_letters' if arm=='qwen35_4b_json' else arm
        if mapped in ALLOWLIST:
            if (r.get('model_id'),r.get('deployment_id'))!=ALLOWLIST[mapped]:raise ValueError('Baseten binding mismatch')
    budget=json.loads(Path(budget_path).read_text())
    amount,wall=validate_budget(budget,summary)
    evidence={}
    private=directory/'freeze-evidence';private.mkdir()
    for name,source in {'audit':audit_path,'context':context_path,'readiness':readiness_path,'budget':budget_path,'analysis':ROOT/'analysis/heldout-analysis-plan.md'}.items():
        destination=private/(name+Path(source).suffix);destination.write_bytes(Path(source).read_bytes())
        evidence[name]={'path':str(destination.relative_to(ROOT)),'sha256':sha(destination)}
    receipt={'summary_sha256':sha(directory/'summary.json'),'evidence':evidence,'reviewer_count':1,'reviewer_id':audit['rater_id'],
             'audit_complete':True,'policy_finalized':True,'context_verified':True,'readiness_verified':True,'billing_verified':True,
             'total_study_planning_ceiling_usd':amount,'max_wall_minutes':wall,
             'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'timestamp_note':'Local freeze timestamp; not a publicly timestamped registration.'}
    atomic(directory/'freeze.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','audit','context','readiness','budget'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy-finalized',action='store_true');a=parser.parse_args()
    print(json.dumps(freeze(a.plan,a.audit,a.context,a.readiness,a.budget,a.policy_finalized),indent=2))
if __name__=='__main__':main()
