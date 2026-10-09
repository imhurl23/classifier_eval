"""Native guard judgments derived from durable predictions, without new calls."""
from run.heldout import derive_native
from run.offline_review import digest
from run.source_full import cohort_key
from run.source_labels import source_scores


def native_results(recorded,rows):
    parents={r['job_id']:r for r in recorded}
    derived=derive_native(recorded,rows)
    for parent in recorded:
        if parent['arm']=='nemotron_guard':
            for target,part in (parent.get('batch_results') or {}).items():
                derived.append({**part,'arm':parent['arm'],'item_id':parent['item_id'],'target':target,'scheme':'s2',
                    'parent_job_ids':[parent['job_id']],'physical_calls':0,'rubric':'native_turn',
                    'scores':source_scores(part,rows[parent['item_id']],target),'model_version':parent['model_version']})
    for result in derived:
        revisions={parents[j]['model_version'] for j in result['parent_job_ids']}
        if len(revisions)!=1:raise ValueError('Native parent model revision mismatch')
        result.update(job_id=digest({'derived':result['parent_job_ids'],'target':result['target']}),
                      cohort=cohort_key(rows[result['item_id']]),rubric=result.get('rubric','native_or'),
                      model_version=revisions.pop(),usage_scope='already_recorded_on_parents')
        result.pop('usage',None)
    if len({r['job_id'] for r in derived})!=len(derived):raise ValueError('Duplicate native derived identity')
    return derived
