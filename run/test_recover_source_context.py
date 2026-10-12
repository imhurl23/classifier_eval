import copy
import pytest
from run.recover_source_context import validate_config


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
