import copy
import os
import pytest
from run.recover_source_context import validate_config
from run import recover_source_context as recovery


def test_context_recovery_rejects_changed_weights_or_generation_stack():
    original={'docker_server':{'start_command':'serve --max-model-len 4096 --dtype bfloat16'},
              'resources':{'accelerator':'L4'},'weights':[{'source':'pinned-revision'}]}
    replacement=copy.deepcopy(original)
    replacement['docker_server']['start_command']='serve --max-model-len 8192 --dtype bfloat16'
    replacement['resources']['accelerator']='L4:2'
    validate_config(original,replacement)
    replacement['weights'][0]['source']='different-revision'
    with pytest.raises(ValueError,match='changes more'):validate_config(original,replacement)
    replacement['weights'][0]['source']='pinned-revision'
    replacement['docker_server']['start_command']+=' --enable-other-decoding'
    with pytest.raises(ValueError,match='changes more'):validate_config(original,replacement)


def test_adapter_reload_keeps_validated_deployment_binding(monkeypatch):
    def old_loader():os.environ['BASETEN_QWEN35_DEPLOYMENT_ID']='old'
    monkeypatch.setattr(recovery.source,'load_local_service_keys',old_loader)
    monkeypatch.setattr(recovery.pilot,'load_local_service_keys',old_loader)
    monkeypatch.setenv('BASETEN_QWEN35_DEPLOYMENT_ID','old')
    recovery.install_bindings({'qwen35_4b_letters':('model','validated')})
    recovery.source.load_local_service_keys()
    assert os.environ['BASETEN_QWEN35_DEPLOYMENT_ID']=='validated'
    recovery.pilot.load_local_service_keys()
    assert os.environ['BASETEN_QWEN35_DEPLOYMENT_ID']=='validated'
