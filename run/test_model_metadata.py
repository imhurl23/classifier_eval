import pytest

from run.model_metadata import experiment_model_metadata, model_metadata
from run.maintain_experiments import span_arm


def test_multi_model_experiment_keeps_arm_mapping_and_deduplicates_models():
    metadata = experiment_model_metadata(['decisions_luna', 'luna_structured', 'jev_choice'])
    assert metadata['models'] == ['gpt-6-luna', 'jev-1.13.0']
    assert metadata['model'] == 'gpt-6-luna | jev-1.13.0'
    assert metadata['multi_model'] is True
    assert metadata['model_by_arm']['decisions_luna'] == metadata['model_by_arm']['luna_structured']


def test_single_model_experiment_groups_as_model_name():
    assert experiment_model_metadata(['jev_choice', 'jev_noul'])['model'] == 'jev-1.13.0'


def test_unknown_arm_cannot_silently_log_without_a_model():
    with pytest.raises(KeyError):
        model_metadata('new_unregistered_arm')
    with pytest.raises(ValueError):
        experiment_model_metadata([])


def test_full_context_backfill_uses_span_name_when_arm_metadata_absent():
    row = {'id': 'random-uuid', 'metadata': {'model_version': 'hf:revision'},
           'span_attributes': {'name': 'qwen35_4b_letters__response_harm__s2'}}
    assert span_arm(row, {'name': 'full_context_smoke'}) == 'qwen35_4b_letters'


def test_ambiguous_historical_span_is_rejected():
    with pytest.raises(ValueError):
        span_arm({'id': 'random-uuid'}, {'name': 'unknown', 'metadata': {'arms': ['jev_choice', 'clef_choice']}})


def test_model_registry_matches_adapter_identifiers():
    from adapters.clef import MODEL_INFO
    from adapters.guards import MODELS
    from adapters.tev1 import MODEL_INFO as TEV_MODELS
    for arm, info in MODEL_INFO.items():
        assert model_metadata(arm)['model'] == info[0]
    for arm, info in MODELS.items():
        assert model_metadata(arm)['model'] == info[1]
    for arm, info in TEV_MODELS.items():
        assert model_metadata(arm)['model'] == info[0]
