"""Consistent model identifiers for experiment grouping and individual spans.

Model is the canonical model name; model_version retains the served revision.
Aggregate run metadata can describe several models. The experiment router only
persists single-arm/configuration metadata on actual Braintrust experiments.
This module never instantiates an adapter or makes inference calls.
"""

ARM_MODELS = {
    'jev_choice': 'jev-1.13.0',
    'jev_noul': 'jev-1.13.0',
    'claude_structured': 'claude-sonnet-5-5',
    'luna_structured': 'gpt-6-luna',
    'decisions_luna': 'gpt-6-luna',
    'clef_choice': 'clef',
    'clef_flash_choice': 'clef-flash',
    'tev1_native': 'togethercomputer/Tev1-4B-experimental',
    'qwen35_4b_letters': 'Qwen/Qwen3.5-4B',
    'qwen35_4b_json': 'Qwen/Qwen3.5-4B',
    'safeguard_120b': 'openai/gpt-oss-safeguard-120b',
    'granite_41_8b': 'ibm-granite/granite-guardian-4.1-8b',
    'qwen3guard_8b': 'Qwen/Qwen3Guard-Gen-8B',
    'nemotron_guard': 'nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3',
    'policylm_17b_prompt': 'musubilabs/policylm-1.7b',
}


def model_metadata(arm):
    return {'arm': arm, 'model': ARM_MODELS[arm]}


def experiment_model_metadata(arms):
    arms = sorted(set(arms))
    if not arms:
        raise ValueError('Experiment model metadata requires at least one arm')
    by_arm = {arm: ARM_MODELS[arm] for arm in arms}
    models = sorted(set(by_arm.values()))
    return {'arms': arms, 'model': ' | '.join(models), 'models': models,
            'model_by_arm': by_arm, 'multi_model': len(models) > 1}
