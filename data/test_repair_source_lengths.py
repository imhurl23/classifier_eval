from copy import deepcopy
from data.repair_source_lengths import repair
from data.build_datasets import count_tokens, interaction_text
from run.test_source_full import sample


def test_length_repair_preserves_human_gold_full_inputs_and_original_record():
    from tokenizers import Tokenizer
    from data.bt_auth import ROOT
    tokenizer=Tokenizer.from_file(str(ROOT/'data/raw/qwen-tokenizer/tokenizer.json'))
    row=sample();row['metadata']['token_len_qwen35']=1
    original=deepcopy(row);fixed=repair(row,tokenizer)
    assert row==original
    assert fixed['id']==row['id'] and fixed['input']==row['input'] and fixed['expected']==row['expected']
    assert fixed['metadata']['token_len_qwen35']==count_tokens(tokenizer,interaction_text(row['input']['prompt'],row['input']['response']))
    assert {k:v for k,v in fixed['metadata'].items() if k!='token_len_qwen35'}=={k:v for k,v in row['metadata'].items() if k!='token_len_qwen35'}
