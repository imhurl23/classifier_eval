"""Write a reviewable held-out launch candidate without executing model calls."""
import hashlib,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from data.build_datasets import normalize_text
from run.matrix import ARMS,OPTIONAL,cells,companion_cells
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    path=ROOT/'data/processed/dis-test.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s]
    paired=[r for r in rows if r['input']['unit']=='conversation']
    overlap=json.loads((ROOT/'data/processed/offline-review/f0c7bbda4b65/private/overlap-report.json').read_text())
    exposed={item for entry in overlap['overlaps'] for item in entry['held_ids']}
    eligible=[r for r in paired if r['id'] not in exposed]
    cells_main=[c for c in cells(True) if c['arm'] not in {'safety_tev','cc_repro'}]
    companions=[c for c in companion_cells(True) if c['arm'] not in {'safety_tev','cc_repro'}]
    wild=[r for r in eligible if r['metadata']['source']=='wildguardtest']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('adapters','prompts','run','scorers') for p in sorted((ROOT/folder).rglob('*')) if p.suffix in {'.py','.md','.yaml'} and '__pycache__' not in p.parts}
    hashes['data/label_map.yaml']=hashlib.sha256((ROOT/'data/label_map.yaml').read_bytes()).hexdigest()
    summary={'status':'candidate_not_preregistered_not_launched','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
             'scope_authorization':'Include Baseten models, plus seven hosted arms and PolicyLM prompt-only supplement',
             'source_sample_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_pairs':len(paired),
             'development_exposed_pairs_quarantined':len(paired)-len(eligible),'candidate_primary_pairs':len(eligible),
             'primary_pairs_by_source':dict(Counter(r['metadata']['source'] for r in eligible)),
             'length_policy':'Preserve all original full conversations. No provisional 1,200-token cutoff applied. Coverage cohorts unresolved until actual serving context checks.',
             'primary_cells':len(cells_main),'companion_cells':len(companions),
             'planned_cell_decisions':len(cells_main)*len(eligible)+len(companions)*len(wild),
             'planned_decisions_note':'Not physical API calls: native derived OR reuses turn decisions; PolicyLM is local/User-only; companion cells use WildGuard pairs.',
             'budget_usd':1000,'development_planning_reserve_usd':811.03392,
             'baseten_preflight_planning_reserve_usd':50,
             'file_sha256':hashes,'primary_matrix':cells_main,'companion_matrix':companions,
             'gates':['Baseten live preflight and full-template context coverage','Final physical-call manifest and conservative cost within remaining budget; Decisions billing unresolved',
                      'Freeze policy/label map/model revisions/analysis plan before any held-out call','Single human reviewer identity and policy audit'],
             'analysis_plan':{'primary':'Source-label strict agreement; invalid/refused/missing outputs are failures.',
                              'secondary':['unsafe recall','false unsafe rate','macro F1 where source gold exists','validity/refusal/coverage rates','paired bootstrap confidence intervals by item, stratified by source','latency/token usage and reconciled costs'],
                              'reviewer_count':1,'inter_rater_agreement':'unmeasured with one reviewer',
                              'human_audit':'Single-reviewer judgments. Separate study-policy agreement from source-label agreement. No AI outputs count as independent human ratings.',
                              'multiclass':'Score only genuinely labeled examples; missing gold stays missing. Report scheme coverage.',
                              'native':'Report native-policy and derived-OR arms separately; retain needs_caution rather than silently collapsing it.',
                              'policylm':'Prompt-only supplement; report documented source exposure. No Assistant/refusal or primary interaction claims.',
                              'controls':'Retain separately reported one-turn, option-order, injection, and withheld-definition controls; final physical-call plan still required.'}}
    version=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest()[:12]
    out=ROOT/'data/processed/heldout-candidate'/version;out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in eligible))
    (out/'exposure-quarantine.json').write_text(json.dumps(sorted({r['id'] for r in paired}- {r['id'] for r in eligible}),indent=2)+'\n')
    print(json.dumps({'candidate':str(out.relative_to(ROOT)),**{k:summary[k] for k in ('original_pairs','candidate_primary_pairs','development_exposed_pairs_quarantined','primary_cells','companion_cells','planned_cell_decisions')}}))
if __name__=='__main__':prepare()
