import json
import pytest
from data.jsonl import read_jsonl
from run.retry_source_errors import prepare_retry


def checkpoint(tmp_path,status='partial',inflight=None):
    rows=[{'job_id':s,'arm':'clef_choice','status':s} for s in ('ok','api_error','invalid_output','refusal','truncated')]
    (tmp_path/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'execution.json').write_text(json.dumps({'status':status,'inflight':inflight or {},'completed_requests':5}))
    return rows


def test_retry_archives_provider_errors_and_preserves_all_model_decisions(tmp_path):
    original=checkpoint(tmp_path)
    receipt=prepare_retry(tmp_path)
    assert receipt['retry_jobs']==1
    from pathlib import Path
    assert read_jsonl(Path(receipt['archive'])/'results.jsonl')==original
    assert read_jsonl(tmp_path/'results.jsonl')==[r for r in original if r['status']!='api_error']
    assert json.loads((tmp_path/'execution.json').read_text())['completed_requests']==4


@pytest.mark.parametrize('status,inflight',[('running',{}),('partial',{'unresolved':{}})])
def test_retry_refuses_active_or_ambiguous_calls(tmp_path,status,inflight):
    original=checkpoint(tmp_path,status,inflight)
    with pytest.raises(ValueError):prepare_retry(tmp_path)
    assert read_jsonl(tmp_path/'results.jsonl')==original
