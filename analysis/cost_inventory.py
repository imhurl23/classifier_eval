"""Inventory recorded usage at saved rates; not a billing reconciliation."""
import json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    rates=json.loads((ROOT/'data/processed/study-reference/00de7797f0a6/pricing.json').read_text())
    usage=defaultdict(lambda:defaultdict(float))
    for directory in ('study-kickoff/0ecb68a006dd','study-reference/00de7797f0a6','study-repeatability/477ac56d553a','study-batching/4424eb16d43d'):
        path=ROOT/'data/processed'/directory/'results.jsonl'
        for line in path.open():
            r=json.loads(line);u=r.get('usage') or {};arm=r['arm'];t=usage[arm]
            t['requests']+=1
            details=u.get('input_tokens_details') or {}
            cached=details.get('cached_tokens',u.get('cache_read_input_tokens',0)) or 0
            write=details.get('cache_write_tokens',u.get('cache_creation_input_tokens',0)) or 0
            # OpenAI detailed cache tokens are subsets of input; Anthropic cache
            # creation/read counts are separate from its uncached input field.
            uncached=u.get('input_tokens',u.get('prompt_tokens',0)) or 0
            if details: uncached=max(0,uncached-cached-write)
            t['uncached_input_tokens']+=uncached;t['cached_input_tokens']+=cached;t['cache_write_tokens']+=write
            t['output_tokens']+=u.get('output_tokens',u.get('completion_tokens',0)) or 0
    known=0
    for arm,t in usage.items():
        rate=rates[arm]
        if rate.get('input') is None:t['recorded_usage_cost_usd']=None;continue
        cost=(t['uncached_input_tokens']*rate['input']+t['cached_input_tokens']*rate.get('cached_input',rate['input'])+
              t['cache_write_tokens']*rate.get('cache_write',rate['input'])+t['output_tokens']*rate['output'])/1e6
        t['recorded_usage_cost_usd']=cost;known+=cost
    result={'saved_rates_checked':rates['checked_date'],'scope':'Recorded usage only; excludes unknown Decisions rates, provider fees, unrecorded retries and Baseten compute. Not actual billed spend.',
            'known_rate_recorded_usage_usd':known,'by_arm':dict(usage),'rates':rates}
    (ROOT/'data/processed/cost-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'known_rate_recorded_usage_usd':known,'unknown_rate_input_tokens':usage['decisions_luna']['uncached_input_tokens']}))
if __name__=='__main__':main()
