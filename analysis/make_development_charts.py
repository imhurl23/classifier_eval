"""Rebuild offline development analysis and standalone charts from checkpoints.

Run: .venv/bin/python -m analysis.make_development_charts
No inference, network assets, source-label changes, or human-label fabrication.
"""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from analysis.braintrust_viz import use_braintrust_theme,make_figure,header,clean_axes,savefig,wilson_interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/output/development-2026-10-07'
STAGES={'reference':'study-reference/00de7797f0a6','repeatability':'study-repeatability/477ac56d553a',
        'batching':'study-batching/4424eb16d43d','schema':'study-kickoff/0ecb68a006dd','policylm':'policylm/20261007T173151Z'}
NAMES={'jev_choice':'Jev Choice','jev_noul':'Jev Noul','decisions_luna':'Decisions Luna',
       'clef_flash_choice':'Clef Flash','clef_choice':'Clef','claude_structured':'Claude structured','luna_structured':'Luna structured'}
SEED=20261007
BOOTSTRAPS=5000


def load(stage):
    directory=ROOT/'data/processed'/STAGES[stage]
    path=directory/'results.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if stage!='policylm':
        execution=json.loads((directory/'execution.json').read_text())
        if execution['status']!='completed' or len(rows)!=execution['planned_requests']:
            raise ValueError(f'{stage} is not complete')
        keys=[r['job_id'] for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate paid-call checkpoint')
    return rows


def export_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def interval(values,seed=SEED,stat=np.mean):
    values=np.asarray(values,dtype=float)
    draws=np.random.default_rng(seed).integers(0,len(values),size=(BOOTSTRAPS,len(values)))
    estimates=stat(values[draws],axis=1)
    return [float(x) for x in np.quantile(estimates,[.025,.975])]


def rate_chart(name,title,subtitle,key,rows,value='rate',lo='lo',hi='hi',max_x=1,xlabel='Percentage of evaluated examples'):
    fig,ax=make_figure(width=12,height=7.4,left=.24,right=.87,top=.79)
    y=np.arange(len(rows));vals=np.array([r[value] for r in rows])
    ax.barh(y,vals,color='#956BDD',edgecolor='white',height=.53,zorder=2)
    lower=np.array([r[lo] for r in rows]);upper=np.array([r[hi] for r in rows])
    ax.errorbar(vals,y,xerr=np.maximum(0,np.vstack([vals-lower,upper-vals])),fmt='none',ecolor='#18181B',capsize=4,lw=1.5,zorder=4)
    ax.set_yticks(y,[NAMES[r['arm']] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,max_x)
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel(xlabel)
    for i,r in enumerate(rows):
        ax.text(1.025,i,f'{r[value]:.1%}  ({r["k"]}/{r["n"]})',transform=ax.get_yaxis_transform(),va='center',fontsize=12,fontweight='bold',clip_on=False)
    clean_axes(ax,despine_left=True,grid_axis='x');header(ax,title,subtitle,key)
    for ext in ('png','svg'):savefig(fig,OUT/f'{name}.{ext}')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);use_braintrust_theme()
    runs={stage:load(stage) for stage in STAGES}
    reference=runs['reference'];lookup={(r['arm'],r['item_id'],r['target']):r for r in reference}
    interaction=defaultdict(list)
    tidy=[]
    for stage,rows in runs.items():
        for r in rows:
            parts=r.get('batch_results') or {r.get('target','prompt_harm'):r}
            for target,p in parts.items():
                tidy.append(dict(stage=stage,arm=r.get('arm','policylm_17b_prompt'),item_id=r['item_id'],target=target,
                                 scheme=r.get('scheme','s2'),rubric=r.get('rubric','prompt_harm_s2'),replicate=r.get('replicate',0),
                                 job_id=r.get('job_id',''),request_sha256=r.get('request_sha256',''),status=p['status'],label=p['label'],
                                 latency_ms=p['latency_ms'],latency_scope='shared bundle' if 'batch_results' in r else 'single request',
                                 model_version=r.get('model_version','hf:4e5a98c579e6ab28c45af42e7d69a84da38f9b12' if stage=='policylm' else ''),source_gold=r.get('human_interaction_gold') if target=='interaction_harm' else r.get('human_prompt_gold') if stage=='policylm' else None))
    export_csv('decisions.csv',tidy)
    for r in reference:
        if r['target']=='interaction_harm':interaction[r['arm']].append(r)
    ids=sorted({r['item_id'] for rows in interaction.values() for r in rows})
    if len(ids)!=200:raise ValueError('Expected exactly 200 matched reference pairs')
    metrics=[];vectors={};golds=None
    for arm,rows in interaction.items():
        by_id={r['item_id']:r for r in rows}
        if set(by_id)!=set(ids):raise ValueError('Unmatched arms')
        rows=[by_id[i] for i in ids]
        gold=np.array([r['human_interaction_gold'] for r in rows]);pred=np.array([r['label'] if r['status']=='ok' else 'failed_output' for r in rows])
        if golds is not None and not np.array_equal(gold,golds):raise ValueError('Gold mismatch')
        golds=gold;correct=(gold==pred).astype(float);vectors[arm]=correct
        unsafe=gold=='unsafe';safe=gold=='safe'
        k=int(correct.sum());lo,hi=wilson_interval(k,len(rows));recall=int(np.sum(pred[unsafe]=='unsafe'));fp=int(np.sum(pred[safe]=='unsafe'))
        f1=[]
        for label in ('safe','unsafe'):
            tp=int(np.sum((gold==label)&(pred==label)));falsep=int(np.sum((gold!=label)&(pred==label)));falsen=int(np.sum((gold==label)&(pred!=label)))
            f1.append(2*tp/(2*tp+falsep+falsen))
        metrics.append(dict(arm=arm,k=k,n=len(rows),rate=k/len(rows),lo=lo,hi=hi,macro_f1=float(np.mean(f1)),
                            unsafe_n=int(unsafe.sum()),unsafe_detected=recall,unsafe_recall=recall/int(unsafe.sum()),
                            unsafe_recall_ci=wilson_interval(recall,int(unsafe.sum())),safe_n=int(safe.sum()),false_unsafe=fp,
                            false_unsafe_rate=fp/int(safe.sum()),false_unsafe_ci=wilson_interval(fp,int(safe.sum())),
                            failed_outputs=int(np.sum(pred=='failed_output'))))
    metrics.sort(key=lambda r:-r['rate'])
    rate_chart('01_source_agreement','Models agree with source labels on 76–82.5% of pairs',
               'Development Aegis pairs · binary interaction harm · long rubric · n = 200 per arm',
               '95% Wilson intervals; failed outputs count as incorrect. Not held-out or audited policy accuracy.',metrics)
    # Matched-item bootstrap, stratified by source safe/unsafe gold; same draw across arms.
    rng=np.random.default_rng(SEED);strata=[np.flatnonzero(golds==g) for g in ('safe','unsafe')]
    draws=np.concatenate([rng.choice(s,size=(BOOTSTRAPS,len(s)),replace=True) for s in strata],axis=1)
    differences=[]
    for a,b in combinations(sorted(vectors),2):
        delta=vectors[a]-vectors[b];lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        differences.append(dict(arm_a=a,arm_b=b,delta_pp=float(delta.mean()*100),lo_pp=float(lo*100),hi_pp=float(hi*100),n=200,
                                a_correct_b_wrong=int(np.sum(delta==1)),b_correct_a_wrong=int(np.sum(delta==-1))))
    export_csv('paired_agreement_differences.csv',differences)
    fig,ax=make_figure(width=12,height=7.5,left=.24,right=.97,top=.76)
    for i,r in enumerate(metrics):
        for value,ci,offset,color in [('unsafe_recall','unsafe_recall_ci',-.14,'#2C1FEB'),('false_unsafe_rate','false_unsafe_ci',.14,'#E6A4D3')]:
            x=r[value];low,high=r[ci]
            ax.errorbar(x,i+offset,xerr=[[max(0,x-low)],[max(0,high-x)]],fmt='o',color=color,ecolor='#18181B',capsize=4,ms=8,zorder=4,
                        label='Unsafe recall (higher is better)' if i==0 and offset<0 else 'False unsafe rate (lower is better)' if i==0 else None)
    ax.set_yticks(range(len(metrics)),[NAMES[r['arm']] for r in metrics]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Rate within each source-label group');clean_axes(ax,True,'x')
    header(ax,'Similar agreement hides different harm tradeoffs','Development pairs · 126 unsafe and 74 safe source labels per arm',
           '95% Wilson intervals. Refusals are missed detections; false unsafe rate counts explicit unsafe predictions.')
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.26),frameon=False,ncol=1,fontsize=11)
    for ext in ('png','svg'):savefig(fig,OUT/f'02_harm_tradeoffs.{ext}')
    plt.close(fig)
    diagnostics={}
    for stage in ('repeatability','batching'):
        groups=defaultdict(lambda:defaultdict(list))
        call_changes=defaultdict(Counter)
        for r in runs[stage]:
            parts=r.get('batch_results') or {r['target']:r}
            for target,p in parts.items():
                base=lookup[r['arm'],r['item_id'],target]
                valid=p['status']=='ok' and base['status']=='ok'
                changed=valid and p['label']!=base['label']
                groups[r['arm']][r['item_id']].append((valid,changed))
                call_changes[r['arm']]['comparisons']+=1
                call_changes[r['arm']]['valid_comparisons']+=valid
                call_changes[r['arm']]['label_changes']+=changed
        aggregate=[]
        for arm,items in groups.items():
            # Treat each item as one trial, retaining all tasks/repeats together.
            full=[parts for parts in items.values() if all(v for v,c in parts)]
            k=sum(any(c for v,c in parts) for parts in full);n=len(full);lo,hi=wilson_interval(k,n)
            aggregate.append(dict(arm=arm,k=k,n=n,rate=k/n,lo=lo,hi=hi,excluded_items=len(items)-n,
                                  **dict(call_changes[arm])))
        aggregate.sort(key=lambda r:(-r['rate'],r['arm']));diagnostics[stage]=aggregate
        if stage=='repeatability':
            title='Most arms repeated their labels on every valid item'
            sub='Any changed label across two extra repeats · 40 development pairs'
            key='95% Wilson intervals; fully valid items only. Noul: 1 task; other arms: 4 tasks per repeat.'
            chart='03_repeatability'
        else:
            title='Bundling changes some Clef and Clef Flash labels'
            sub='Any changed label across three bundled companion tasks · 40 pairs per arm'
            key='95% Wilson intervals; compare against matched separate requests, with identical option order.'
            chart='04_batching'
        rate_chart(chart,title,sub,key,aggregate,max_x=.5)
    reliability=[]
    for arm in NAMES:
        rows=[r for r in reference if r['arm']==arm];by_item=defaultdict(list)
        for r in rows:by_item[r['item_id']].append(float(r['status']!='ok'))
        failures=sum(r['status']!='ok' for r in rows)
        # Correlated companion tasks are resampled together, not as 800 independent trials.
        means=[np.mean(v) for v in by_item.values()];lo,hi=interval(means)
        reliability.append(dict(arm=arm,k=failures,n=len(rows),rate=failures/len(rows),lo=lo,hi=hi,
                                statuses=dict(Counter(r['status'] for r in rows))))
    reliability.sort(key=lambda r:(-r['rate'],r['arm']))
    rate_chart('05_output_reliability','Reference failures were Claude refusals',
               'Completed reference requests · 800 per common arm, 200 for Noul',
               '95% item-cluster bootstrap intervals; zero failures give degenerate observed-data bootstrap intervals.',reliability,max_x=.05,xlabel='Percentage of reference requests')
    # Single-question interaction latency only; never mix shared bundle timings or repeats.
    latency=[]
    for arm,rows in interaction.items():
        values=np.array([r['latency_ms']/1000 for r in rows]);lo,hi=interval(values,stat=np.median)
        latency.append(dict(arm=arm,median_seconds=float(np.median(values)),lo_seconds=lo,hi_seconds=hi,
                            p95_seconds=float(np.quantile(values,.95)),n=len(values)))
    latency.sort(key=lambda r:r['median_seconds'])
    fig,ax=make_figure(width=12,height=7.4,left=.24,right=.87,top=.79)
    for i,r in enumerate(latency):
        x=r['median_seconds'];ax.errorbar(x,i,xerr=[[max(0,x-r['lo_seconds'])],[max(0,r['hi_seconds']-x)]],fmt='o',color='#2C1FEB',ecolor='#18181B',capsize=4,ms=9)
        ax.text(1.025,i,f'{x:.3f}s',transform=ax.get_yaxis_transform(),va='center',fontsize=12,fontweight='bold')
    ax.set_yticks(range(len(latency)),[NAMES[r['arm']] for r in latency]);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Median end-to-end request latency (seconds, log scale)');clean_axes(ax,True,'x')
    header(ax,'Observed request latency varies across model systems','Reference interaction-harm requests only · n = 200 per arm',
           '95% item bootstrap intervals. Includes network/provider effects; not an isolated model-speed benchmark.')
    for ext in ('png','svg'):savefig(fig,OUT/f'06_latency.{ext}')
    plt.close(fig)
    labeled=[r for r in runs['policylm'] if r.get('human_prompt_gold') is not None]
    policy={'total_predictions':len(runs['policylm']),'valid':sum(r['status']=='ok' for r in runs['policylm']),
            'human_prompt_labeled_n':len(labeled),'source_correct':sum(r['status']=='ok' and r['label']==r['human_prompt_gold'] for r in labeled),
            'note':'User-only supplement. Eight reserved WildGuard pairs; documented source exposure. Other 200 User judgments lack independent human prompt gold.'}
    summary={'scope':'Completed development data only; no held-out conclusions or completed human policy audit.',
             'seed':SEED,'bootstrap_samples':BOOTSTRAPS,'reference':metrics,'repeatability':diagnostics['repeatability'],
             'batching':diagnostics['batching'],'reliability':reliability,'latency':latency,'paired_differences':differences,'policylm':policy,
             'schema_statuses':dict(Counter(r['arm']+'|'+r['status'] for r in runs['schema'])),
             'source_file_sha256':{stage:hashlib.sha256((ROOT/'data/processed'/path/'results.jsonl').read_bytes()).hexdigest() for stage,path in STAGES.items()}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    export_csv('reference_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict))} for r in metrics])
    export_csv('latency_metrics.csv',latency)
    report=['# Development analysis — 2026-10-07','',
            'The completed development runs support technical and exploratory comparisons. The main reference sample contains 200 Aegis V2 validation conversation pairs (126 unsafe, 74 safe); every hosted arm classified the same pairs with the binary long rubric. This is source-label agreement, not independently reviewed study-policy accuracy or held-out generalization. The stratified sample and descriptive intervals do not establish population prevalence.','',
            '| Model | Strict source agreement | Macro F1 | Unsafe recall | False unsafe rate |',
            '|---|---:|---:|---:|---:|']
    for r in metrics:report.append(f'| {NAMES[r["arm"]]} | {r["rate"]:.1%} | {r["macro_f1"]:.3f} | {r["unsafe_recall"]:.1%} | {r["false_unsafe_rate"]:.1%} |')
    report+=['','![Source-label agreement](01_source_agreement.png)','','![Harm tradeoffs](02_harm_tradeoffs.png)','',
             'The observed agreement range is 76–82.5%. Avoid a definitive winner claim from this development sample. The paired-difference table resamples matching items within safe/unsafe strata (5,000 draws, fixed seed). Its 21 pairwise intervals are exploratory, nominal 95% intervals with no multiple-comparison adjustment. Overlap of individual model intervals is not itself a paired significance test.','',
             '## Repeatability and batching','',
             '| Model | Repeat label changes / valid comparisons | Items with any repeat change | Items excluded for failed outputs |',
             '|---|---:|---:|---:|']
    for r in diagnostics['repeatability']:report.append(f'| {NAMES[r["arm"]]} | {r["label_changes"]}/{r["valid_comparisons"]} | {r["k"]}/{r["n"]} | {r["excluded_items"]} |')
    report+=['','![Repeatability](03_repeatability.png)','',
             'Repeatability uses two extra calls per task on 40 reference pairs. Items with any failed reference/repeat output are excluded from the any-change chart and counted separately. Noul has one task; other arms have four, so the item-level chance of a change is not directly comparable across those task counts. Zero observed changes do not establish universal determinism.','',
             '| Model | Batched label changes / comparisons | Items with any batching change |',
             '|---|---:|---:|']
    for r in diagnostics['batching']:report.append(f'| {NAMES[r["arm"]]} | {r["label_changes"]}/{r["valid_comparisons"]} | {r["k"]}/{r["n"]} |')
    report+=['','![Batching](04_batching.png)','',
             'All 160 batched calls produced 480 valid decisions. Batching changed 5/120 Clef decisions and 9/120 Clef Flash decisions, versus 0/120 for Jev Choice and Decisions Luna. The companion tasks do not have independent human gold in this Aegis development sample; changes measure disagreement between request forms, not improvement or degradation in accuracy. Retain separate requests for the primary matched comparison.','',
             '## Reliability and latency','',
             'The reference run completed 5,000 calls: 4,988 valid outputs and 12 Claude refusals. Repeatability completed 2,000 calls: 1,984 valid outputs and 16 Claude refusals. Neither stage had API errors. The earlier schema check contains three invalid Jev distributions and five Decisions refusals; those failures are retained separately rather than hidden in a binary reference aggregate.','',
             '![Output reliability](05_output_reliability.png)','','![Latency](06_latency.png)','',
             'Latency is the observed end-to-end adapter timing for reference interaction requests, including provider/network and any retries. It is not isolated inference throughput; these arms have different providers and interfaces. Failure intervals resample entire items to preserve correlation across tasks. Bootstrap intervals for zero observed failures are degenerate; they are not a guarantee of zero future failure risk.','',
             '## PolicyLM supplement and remaining limits','',
             f'PolicyLM returned valid outputs for {policy["valid"]}/{policy["total_predictions"]} User-only inputs. It matched prompt source labels on {policy["source_correct"]}/{policy["human_prompt_labeled_n"]} reserved WildGuard pairs. The tiny source-exposed subset is not a reliable generalization comparison and is excluded from primary model rankings. The other 200 User predictions have no independent human prompt gold.','',
             'Baseten models have no successful new preflight in this stage; Safeguard readiness timed out and all six deployments were shut down. No Baseten performance ranking is inferred. The audit is configured for one reviewer, but actual judgments are still pending. Human agreement statistics remain unmeasured. Exact dollar comparisons are omitted because Decisions rates, provider fees, and compute billing are unresolved.','',
             'Recreate with `.venv/bin/python -m analysis.make_development_charts`. `decisions.csv` contains tidy decision-level records without raw conversation/span payloads. `summary.json` includes input hashes, counts, and methods. Each chart is exported as PNG and SVG.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'output':str(OUT),'charts':6,'policylm':policy,'reference':[(r['arm'],r['rate']) for r in metrics]},indent=2))

if __name__=='__main__':main()
