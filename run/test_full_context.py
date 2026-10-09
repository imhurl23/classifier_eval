import json
from pathlib import Path
import pytest
from run.full_context import request
from run.full_development import budget
from data.full_conversations import ROOT


def row():
    return json.loads((ROOT/'data/processed/full-conversations/92f8046d8b14/development.jsonl').read_text().splitlines()[0])


def test_entire_numbered_history_is_transmitted_to_direct_decisions():
    r=row();wire=request('decisions_luna',r,'response_harm','response_harm_s2','s2')
    assert wire['input']==r['input']['text']
    assert all(m['content'] in wire['input'] for m in r['input']['messages'])
    assert r['input']['target_assistant_turn_id'] in str(wire)


def test_pair_only_or_truncated_history_is_rejected():
    r=row();r['input']['text']=r['input']['prompt']+'\n'+r['input']['response']
    with pytest.raises(ValueError,match='omitted'):request('jev_choice',r,'interaction_harm','r_long','s2')


def test_native_final_pair_fallback_is_not_allowed():
    with pytest.raises(ValueError,match='Native guard'):request('qwen3guard_8b',row(),'interaction_harm','r_long','s2')


def test_full_reference_budget_cannot_silently_exceed_existing_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr('run.full_development.ROOT',tmp_path)
    with pytest.raises(ValueError,match='authorized'):budget(tmp_path,{'uniform_three_attempt_planning_reserve_usd':200})
    prior,reserve=budget(tmp_path,{'uniform_three_attempt_planning_reserve_usd':20})
    assert prior+reserve<1000
