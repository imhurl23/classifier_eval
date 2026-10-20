from copy import deepcopy

from run.luna_explicit_categories import variant_request, plan


def test_explicit_variant_changes_only_instructions_without_mutating_original():
    original = {'model': 'gpt-6-luna', 'input': 'User: example\nAssistant: reply',
                'questions': [{'name': 'safety', 'type': 'choice', 'instructions': 'Original',
                               'choices': [{'value': 'unsafe', 'description': 'Unsafe'},
                                           {'value': 'safe', 'description': 'Safe'}]}]}
    frozen = deepcopy(original)
    explicit = variant_request(original, 'explicit_categories', 'Extra rules')
    assert original == frozen
    assert explicit['questions'][0]['instructions'] == 'Original\n\nExtra rules'
    explicit['questions'][0]['instructions'] = 'Original'
    assert explicit == original
    assert variant_request(original, 'original_policy_repeat', 'Extra rules') == original


def test_plan_keeps_all_unsafe_and_safe_controls_paired_without_category_leakage():
    jobs = plan()
    assert len(jobs) == len({j['job_id'] for j in jobs}) == 3362
    assert sum(j['source_gold'] == 'safe' for j in jobs) == 936 * 2
    assert sum(j['source_gold'] == 'unsafe' for j in jobs) == 745 * 2
    instructions = set()
    by_item = {}
    for job in jobs:
        by_item.setdefault(job['item_id'], {})[job['variant']] = job
        if job['variant'] == 'explicit_categories':
            instructions.add(job['request']['questions'][0]['instructions'])
    assert len(instructions) == 1  # Same complete rules for every input; no category hint.
    for pair in by_item.values():
        original, explicit = pair['original_policy_repeat'], pair['explicit_categories']
        assert original['request']['input'] == explicit['request']['input']
        assert original['request']['questions'][0]['choices'] == explicit['request']['questions'][0]['choices']
