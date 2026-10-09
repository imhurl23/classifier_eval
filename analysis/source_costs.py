"""Recorded hosted-usage estimate; reservations and invoices remain separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(usage, rate):
    total = usage.get('input_tokens', usage.get('prompt_tokens'))
    if total is None:
        return None
    details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
    # OpenAI reports cache counts within input; Anthropic reports them separately.
    cached = details.get('cached_tokens',0) or 0
    written = details.get('cache_write_tokens',0) or 0
    plain = max(0,total-cached-written)
    cached += usage.get('cache_read_input_tokens',0) or 0
    written += usage.get('cache_creation_input_tokens',0) or 0
    output = usage.get('output_tokens',usage.get('completion_tokens',0)) or 0
    return (plain*rate['input'] + cached*rate.get('cached_input',rate['input']) +
            written*rate.get('cache_write',rate['input']) + output*rate['output'])/1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rates = json.loads((ROOT/'analysis/hosted-pricing-2026-10-09.json').read_text())
    groups = defaultdict(lambda:{'requests':0,'usage_available':0,'estimated_usd':0,'statuses':defaultdict(int)})
    # A running process may be appending its last record; only completed lines
    # are included. Successful calls are not inferred from planning allocations.
    records = [json.loads(line) for line in args.results.read_bytes().split(b'\n')[:-1] if line]
    if len({r['job_id'] for r in records}) != len(records):
        raise ValueError('Duplicate physical requests')
    for row in records:
        group=groups[row['arm']];group['requests']+=1;group['statuses'][row['status']]+=1
        cost=estimate(row.get('usage') or {},rates[row['arm']])
        if cost is not None:
            group['usage_available']+=1;group['estimated_usd']+=cost
    report={'scope':str(args.results),'rates_checked':rates['checked_date'],'arms':dict(groups),
            'estimated_recorded_usage_usd':sum(g['estimated_usd'] for g in groups.values()),
            'billing_reconciled':False,
            'limitations':'Public standard rates and recorded usage only; excludes unrecorded retry usage, gateway fees, regional/long-context premiums, Cloudflare subscription and Baseten compute. Reservations are not spend.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
