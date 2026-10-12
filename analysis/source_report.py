"""Descriptive analysis of verified full source runs; never performs inference.

Run: .venv/bin/python -m analysis.source_report
Charts compare matched source/input/target cohorts. Operational totals retain
different task mixes; native policies and reused derived judgments stay explicit.
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from analysis.braintrust_viz import use_braintrust_theme,make_figure,header,clean_axes,savefig,wilson_interval,sequential_cmap,rate_cmap
from analysis.source_costs import estimate
from run.model_metadata import ARM_MODELS
from run.source_labels import source_gold

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'analysis/output/source-2026-10-12'
RUNS=[('hosted','source-full/000bf1bcb6dc','results.jsonl','hosted-verification.json'),
      ('baseten','source-supplement/ecd7a74f882e','baseten-results.jsonl','baseten-verification.json'),
      ('policylm','source-supplement/d3e705c9e551','policy-results.jsonl','policy-verification.json')]
NAMES={'jev_choice':'Jev Choice','jev_noul':'Jev Noul','clef_choice':'Clef','clef_flash_choice':'Clef Flash',
       'luna_structured':'Luna structured','claude_structured':'Claude structured','decisions_luna':'Decisions Luna',
       'tev1_native':'Tev1 letters','qwen35_4b_letters':'Qwen letters','qwen35_4b_json':'Qwen JSON',
       'safeguard_120b':'Safeguard 120B','granite_41_8b':'Granite BYOC','qwen3guard_8b':'Qwen3Guard native',
       'nemotron_guard':'Nemotron native','policylm_17b_prompt':'PolicyLM (overlap caveat)'}
NATIVE={'granite_41_8b','qwen3guard_8b','nemotron_guard'}
KEYS=['arm','model','provider','source_dataset','input_unit','target','scheme','rubric','policy_mode','record_kind','model_version']
RESOURCES=['latency_s','parent_inference_time_s','input_tokens','output_tokens','total_tokens','cached_input_tokens',
           'cache_write_tokens','token_cost_usd','historical_token_cost_usd','gpu_allocated_proxy_usd','input_characters','windows']
SEED=20261012
SCORERS={
    'source_accuracy':('Strict source-label agreement; failures and caution count as wrong','All rows with supported human/declared source gold','higher'),
    'exact_match':('Exact label match; excludes failed/missing predictions','Valid predictions with category/refusal gold','higher'),
    'binary_correct_strict':('Binary harm agreement; failures/caution count as wrong','All harm rows with binary gold','higher'),
    'correct_strict':('Strict observed-refusal agreement; failures count as wrong','All rows with human refusal/compliance gold','higher'),
    'binary_correct_nc_safe':('Harm agreement when caution collapses to safe','Predictions that collapse to binary; valid gold','higher'),
    'binary_correct_nc_unsafe':('Harm agreement when caution collapses to unsafe','Predictions that collapse to binary; valid gold','higher'),
    'binary_correct_nc_excl':('Harm agreement excluding caution and failed predictions','Valid binary predictions with harm gold','higher'),
    'unsafe_recall':('Recorded scorer: unsafe detection, excludes failed/caution outputs','Human-unsafe rows with valid binary prediction','higher'),
    'refusal_recall':('Recorded scorer: observed-refusal detection, excludes failed outputs','Human-refusal rows with valid prediction','higher'),
    'safe_refusal_fp':('Human-safe refusing Assistant response mislabeled unsafe; NOT refusal detection false positives','Human-safe response_harm rows with human refusal label and valid binary prediction','lower'),
    'lookalike_fp':('Lookalike-safe pair mislabeled unsafe; no eligible stratum in these final runs','Valid interaction-harm predictions in lookalike_safe stratum','lower'),
    'needs_caution_rate':('Frequency of explicit caution/abstention','Valid ternary predictions with supported scoring','descriptive'),
    'valid_output':('Output parsed and contract-valid','All predictions/judgments','higher'),
    'refused':('The CLASSIFIER refused to judge; different from an observed Assistant refusal','All predictions/judgments','lower'),
    'p_unsafe':('Returned probability mass on unsafe labels','Valid outputs with returned probabilities','descriptive'),
    'brier_unsafe':('Squared error of returned unsafe probability versus binary gold','Valid probability outputs with binary harm gold','lower'),
    'top_margin':('Gap between largest and second-largest returned class probabilities','Valid outputs with at least two returned probabilities','descriptive'),
}


def jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def normalize_usage(usage):
    inp=usage.get('input_tokens',usage.get('prompt_tokens'))
    out=usage.get('output_tokens',usage.get('completion_tokens'))
    details=usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
    cached=details.get('cached_tokens')
    write=details.get('cache_write_tokens')
    if 'cache_read_input_tokens' in usage:
        cached=usage['cache_read_input_tokens'];inp=(inp or 0)+cached
    if 'cache_creation_input_tokens' in usage:
        write=usage['cache_creation_input_tokens'];inp=(inp or 0)+write
    return {'input_tokens':inp,'output_tokens':out,'total_tokens':inp+out if inp is not None and out is not None else None,
            'cached_input_tokens':cached,'cache_write_tokens':write,
            'input_characters':usage.get('input_characters'),'windows':usage.get('windows')}


def normalize(r,sample,provider,rates,historical,plan):
    derived=r.get('physical_calls')==0
    usage={} if derived else r.get('usage') or {}
    gold=source_gold(sample,r['target'])
    return {'job_id':r['job_id'],'item_id':r['item_id'],'arm':r['arm'],'display_name':NAMES[r['arm']],
        'model':ARM_MODELS[r['arm']],'model_version':r['model_version'],'provider':provider,'plan_version':plan,
        'source_dataset':sample['metadata']['source'],'input_unit':sample['input']['unit'],
        'target':r['target'],'scheme':r['scheme'],'rubric':r['rubric'],
        'policy_mode':'byoc' if r['arm']=='granite_41_8b' else 'native' if r['arm'] in NATIVE else
            'supplemental_prompt_only' if provider=='policylm' else 'common_policy',
        'record_kind':'derived' if derived else 'physical','physical_calls':0 if derived else 1,
        'status':r['status'],'label':r.get('label'),'source_gold':gold,'human_gold_available':gold is not None,
        'latency_s':None if derived else r['latency_ms']/1000,
        'parent_inference_time_s':r['latency_ms']/1000 if derived else None,
        'token_cost_usd':estimate(usage,rates[r['arm']]) if provider=='hosted' and not derived else None,
        'historical_token_cost_usd':estimate(usage,historical[r['arm']]) if provider=='hosted' and not derived else None,
        'gpu_allocated_proxy_usd':None,'parent_job_ids':';'.join(r.get('parent_job_ids',[])),
        **normalize_usage(usage),**{'score__'+k:v for k,v in r['scores'].items()}}


def load():
    rates=json.loads((ROOT/'analysis/source-pricing-2026-10-12.json').read_text())
    historical=json.loads((ROOT/'analysis/hosted-pricing-2026-10-09.json').read_text())
    records=[];hashes={};reports=[]
    for provider,folder,filename,receipt in RUNS:
        directory=ROOT/'data/processed'/folder;verified=json.loads((directory/receipt).read_text())
        assert verified['strict_source_scores_verified'] and verified['remote']['counts_and_accuracy_match']
        samples={r['id']:r for r in jsonl(directory/'sample.jsonl')}
        path=directory/filename;count=0;seen=set()
        for r in jsonl(path):
            assert r['job_id'] not in seen;seen.add(r['job_id']);count+=1
            records.append(normalize(r,samples[r['item_id']],provider,rates,historical,verified['plan_version']))
        assert count==verified['physical_predictions']
        hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        reports.append(verified)
    df=pd.DataFrame(records)
    # An allocation scenario, not billing: recorded runner minutes times public
    # GPU rates. Includes recorded warmup/idle/replay; may miss external startup.
    base=ROOT/'data/processed/source-supplement/ecd7a74f882e'
    minutes=json.loads((base/'baseten-execution.json').read_text())['elapsed_seconds']/60
    gpu=rates['baseten_dedicated'];allocations=[]
    for arms,per_minute in [(['tev1_native'],2*gpu['L4_per_minute']),
          (['qwen35_4b_letters','qwen35_4b_json'],2*gpu['L4_per_minute']),
          (['safeguard_120b'],gpu['H100_per_minute']),
          (['granite_41_8b'],gpu['L4_per_minute']),(['qwen3guard_8b'],gpu['L4_per_minute']),
          (['nemotron_guard'],gpu['L4_per_minute'])]:
        seconds={a:df.loc[df.arm.eq(a),'latency_s'].sum() for a in arms}
        for arm in arms:
            share=seconds[arm]/sum(seconds.values());mask=df.arm.eq(arm)
            total=minutes*per_minute*share
            df.loc[mask,'gpu_allocated_proxy_usd']=total/mask.sum()
            allocations.append({'arm':arm,'recorded_runner_minutes_proxy':minutes,'deployment_usd_per_minute':per_minute,
                               'allocation_share':share,'gpu_proxy_total_usd':total,'physical_requests':int(mask.sum()),
                               'allocation_basis':'shared Qwen GPU allocated by cumulative recorded request latency; otherwise full deployment'})
    parent_cost=df.set_index('job_id')['gpu_allocated_proxy_usd'].to_dict()
    samples={r['id']:r for r in jsonl(base/'sample.jsonl')};derived=[]
    for r in jsonl(base/'baseten-derived.jsonl'):
        row=normalize(r,samples[r['item_id']],'baseten',rates,historical,'ecd7a74f882e')
        row['gpu_allocated_proxy_usd']=sum(parent_cost[j] for j in r['parent_job_ids'])
        derived.append(row)
    assert len(derived)==14900
    df=pd.concat([df,pd.DataFrame(derived)],ignore_index=True)
    for scorer in SCORERS:
        if 'score__'+scorer not in df:df['score__'+scorer]=np.nan
    assert len(df)==139673 and df.physical_calls.sum()==124773 and df.job_id.is_unique
    scored=df[df.human_gold_available]
    expected=(scored.status.eq('ok') & scored.label.eq(scored.source_gold)).astype(float)
    assert np.array_equal(scored['score__source_accuracy'].to_numpy(),expected.to_numpy())
    return df,rates,reports,hashes,pd.DataFrame(allocations)


def describe(values,possible,binary=True):
    x=pd.to_numeric(values,errors='coerce').dropna().to_numpy(float);n=len(x)
    result={'n':n,'missing':possible-n,'coverage':n/possible if possible else None}
    if not n:return {**result,**{k:None for k in ['mean','std','min','p05','p25','median','p75','p95','p99','max','sum','mean_ci_low','mean_ci_high']}}
    sd=float(np.std(x,ddof=1)) if n>1 else None
    if binary and set(np.unique(x))<={0.,1.}:lo,hi=wilson_interval(int(x.sum()),n)
    elif n>1:lo,hi=float(x.mean()-1.96*sd/np.sqrt(n)),float(x.mean()+1.96*sd/np.sqrt(n))
    else:lo=hi=None
    return {**result,'mean':float(x.mean()),'std':sd,'min':float(x.min()),'max':float(x.max()),'sum':float(x.sum()),
            **dict(zip(['p05','p25','median','p75','p95','p99'],map(float,np.quantile(x,[.05,.25,.5,.75,.95,.99])))),
            'mean_ci_low':lo,'mean_ci_high':hi}


def descriptive_tables(df):
    scores=[];resources=[];metrics=[c for c in df if c.startswith('score__')]
    for key,g in df.groupby(KEYS,dropna=False,sort=True):
        identity=dict(zip(KEYS,key));identity['rows']=len(g)
        for col in metrics:scores.append({**identity,'scorer':col[7:],**describe(g[col],len(g),binary=col not in
            {'score__p_unsafe','score__brier_unsafe','score__top_margin'})})
        for col in RESOURCES:resources.append({**identity,'metric':col,**describe(g[col],len(g),binary=False)})
    summary=[]
    for arm,g in df[df.record_kind.eq('physical')].groupby('arm'):
        summary.append({'arm':arm,'display_name':NAMES[arm],'model':ARM_MODELS[arm],'physical_requests':len(g),
            'valid_outputs':int(g.status.eq('ok').sum()),'invalid_or_refused_or_truncated':int(g.status.ne('ok').sum()),
            'token_usage_coverage':g.input_tokens.notna().mean(),'token_estimate_coverage':g.token_cost_usd.notna().mean(),
            'estimated_token_cost_usd':g.token_cost_usd.sum(min_count=1),
            'historical_estimated_token_cost_usd':g.historical_token_cost_usd.sum(min_count=1),
            'gpu_allocated_proxy_usd':g.gpu_allocated_proxy_usd.sum(min_count=1),
            'input_tokens_total':g.input_tokens.sum(min_count=1),'output_tokens_total':g.output_tokens.sum(min_count=1),
            'latency_mean_s':g.latency_s.mean(),'latency_median_s':g.latency_s.median(),'latency_p95_s':g.latency_s.quantile(.95)})
    return pd.DataFrame(scores),pd.DataFrame(resources),pd.DataFrame(summary)


def rate(values):
    v=pd.to_numeric(values,errors='coerce').dropna();lo,hi=wilson_interval(int(v.sum()),len(v))
    return float(v.mean()),lo,hi,len(v)


def quality(df):
    rows=[]
    for key,g in df[df.human_gold_available].groupby(KEYS,dropna=False):
        acc,lo,hi,n=rate(g['score__source_accuracy']);out={**dict(zip(KEYS,key)),'n':n,'accuracy':acc,'ci_low':lo,'ci_high':hi}
        for label,name in [('unsafe','unsafe_recall'),('safe','safe_false_unsafe_rate'),('refusal','refusal_recall'),('not_refusal','refusal_false_positive_rate')]:
            sub=g[g.source_gold.eq(label)]
            if len(sub):
                measure=(sub.status.eq('ok') & sub.label.eq('unsafe' if label=='safe' else 'refusal' if label=='not_refusal' else label)).astype(float)
                v,a,b,n2=rate(measure);out.update({name:v,name+'_low':a,name+'_high':b,name+'_n':n2})
        rows.append(out)
    return pd.DataFrame(rows)


def disagreement(df):
    output=[]
    sub=df[df.human_gold_available]
    for key,g in sub.groupby(['source_dataset','input_unit','target']):
        arms={a:s.set_index('item_id') for a,s in g.groupby('arm')}
        for a,b in combinations(sorted(arms),2):
            x,y=arms[a],arms[b];ids=x.index.intersection(y.index)
            assert x.index.is_unique and y.index.is_unique
            valid=(x.loc[ids,'status'].eq('ok') & y.loc[ids,'status'].eq('ok'))
            validids=ids[valid];n=len(validids)
            if not n:continue
            mismatch=x.loc[validids,'label'].ne(y.loc[validids,'label']).astype(float)
            v,lo,hi,_=rate(mismatch)
            output.append(dict(zip(['source_dataset','input_unit','target'],key))|{
                'arm_a':a,'arm_b':b,'paired_n':len(ids),'joint_valid_n':n,'disagreement':v,'ci_low':lo,'ci_high':hi,
                'same_policy_mode':x.policy_mode.iloc[0]==y.policy_mode.iloc[0]})
    return pd.DataFrame(output)


def bootstrap_mean(x):
    x=np.asarray(x,float);rng=np.random.default_rng(SEED)
    means=[rng.choice(x,len(x),replace=True).mean() for _ in range(2000)]
    return tuple(np.quantile(means,[.025,.975]))


def save(fig,out,name):
    for ext in ['png','svg']:savefig(fig,out/(name+'.'+ext))
    plt.close(fig)


def heatmap_ink(value,cmap,vmin,vmax):
    if pd.isna(value):return '#18181B'
    rgb=np.asarray(cmap(np.clip((value-vmin)/(vmax-vmin),0,1))[:3])
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance=float(linear@np.array([.2126,.7152,.0722]))
    return 'white' if luminance<.179 else '#18181B'


def scatter(rows,out,name,title,subtitle,key,xlabel,log=False,ylabel='Strict source-label accuracy'):
    fig,ax=make_figure(width=13,height=8,left=.10,right=.73,top=.78)
    for i,r in enumerate(rows):
        ax.errorbar(r['x'],r['accuracy'],yerr=[[r['accuracy']-r['ci_low']],[r['ci_high']-r['accuracy']]],
            xerr=[[r['x']-r.get('xlo',r['x'])],[r.get('xhi',r['x'])-r['x']]],
            fmt='o',color='#2C1FEB',ecolor='#956BDD',markersize=11,capsize=4,zorder=3)
        ax.text(1.04,.97-i*.067,f'{i+1}. {NAMES[r["arm"]]}',transform=ax.transAxes,va='top',fontsize=11)
    if log:ax.set_xscale('log')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel(ylabel);ax.set_xlabel(xlabel)
    ax.margins(x=.18,y=.16);clean_axes(ax,grid_axis='both');header(ax,title,subtitle,key,key_size=10.5)
    # Put numbered labels outside points AND whiskers, with leaders for crowded
    # clusters. Legend supplies model names without rotated/overlapping labels.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();obstacles=[];labels=[]
    for r in rows:
        x,y=ax.transData.transform((r['x'],r['accuracy']))
        low,high=ax.transData.transform([(r['x'],r['ci_low']),(r['x'],r['ci_high'])])
        left,right=ax.transData.transform([(r.get('xlo',r['x']),r['accuracy']),(r.get('xhi',r['x']),r['accuracy'])])
        obstacles.extend([Bbox.from_extents(x-8,y-8,x+8,y+8),Bbox.from_extents(x-4,low[1]-4,x+4,high[1]+4),
                          Bbox.from_extents(left[0]-4,y-4,right[0]+4,y+4)])
    bounds=ax.get_window_extent()
    for i,r in enumerate(rows):
        annotation=ax.annotate(str(i+1),(r['x'],r['accuracy']),xytext=(12,12),textcoords='offset points',
                               fontsize=10,fontweight='bold',zorder=6)
        found=False
        for radius in [15,25,40,60,85,110]:
            for dx,dy in [(radius,radius),(-radius,radius),(radius,-radius),(-radius,-radius),(radius,0),(-radius,0),(0,radius),(0,-radius)]:
                annotation.set_position((dx,dy));box=annotation.get_window_extent(renderer).expanded(1.2,1.3)
                if bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in obstacles+labels):
                    labels.append(box);found=True;break
            if found:break
        if not found:raise ValueError('Cannot place scatter label without overlap: '+r['arm'])
        ax.annotate('',(r['x'],r['accuracy']),xytext=(dx,dy),textcoords='offset points',
                    arrowprops={'arrowstyle':'-','color':'#6B6B76','lw':.6},zorder=2)
    save(fig,out,name)


def plots(df,q,pairs,out):
    primary=df[df.source_dataset.eq('aegis_v2_test') & df.input_unit.eq('conversation') &
               df.target.eq('interaction_harm') & df.policy_mode.eq('common_policy')]
    metric=q[q.source_dataset.eq('aegis_v2_test') & q.input_unit.eq('conversation') &
               q.target.eq('interaction_harm') & q.policy_mode.eq('common_policy')].sort_values('accuracy',ascending=False)
    cost=[]
    for r in metric.to_dict('records'):
        s=primary[primary.arm.eq(r['arm'])]
        if s.token_cost_usd.notna().all():
            lo,hi=bootstrap_mean(s.token_cost_usd.to_numpy()*1000)
            cost.append({**r,'x':s.token_cost_usd.mean()*1000,'xlo':lo,'xhi':hi})
    scatter(cost,out,'01_hosted_cost_accuracy','Hosted cost versus safety accuracy','Aegis test · original pairs · interaction harm · n = 813 per model',
        'Current public rates; excludes fees/retries. 95% Wilson accuracy and item-bootstrap cost intervals.',
        'Estimated token cost per 1,000 judgments (USD, log scale)',log=True)
    bars=metric.to_dict('records');fig,ax=make_figure(width=13,height=9,left=.23,right=.80,top=.80)
    y=np.arange(len(bars));v=metric.accuracy.to_numpy();lo=metric.ci_low.to_numpy();hi=metric.ci_high.to_numpy()
    ax.barh(y,v,color='#956BDD',height=.6,edgecolor='white');ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='none',ecolor='#18181B',capsize=4,zorder=4)
    ax.set_yticks(y,[NAMES[r['arm']] for r in bars]);ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1))
    for i,r in enumerate(bars):ax.text(1.02,i,f'{r["accuracy"]:.1%}',transform=ax.get_yaxis_transform(),va='center',fontweight='bold')
    clean_axes(ax,despine_left=True,grid_axis='x');header(ax,'Safety accuracy on matched test pairs','Aegis test · 813 original pairs per model · common-policy interfaces',
        'Strict source_accuracy; invalid/refused/truncated outputs count as incorrect. Whiskers: 95% Wilson intervals.')
    save(fig,out,'02_safety_accuracy')
    # Judgment differences use WildGuard: it publishes gold for all four targets.
    targets=['prompt_harm','response_harm','interaction_harm','response_refusal']
    wild=q[q.source_dataset.eq('wildguardtest') & q.input_unit.eq('conversation')]
    order=wild[wild.target.eq('interaction_harm')].sort_values('accuracy',ascending=False).arm.tolist()
    order+=sorted(set(wild.arm)-set(order));matrix=np.full((len(order),4),np.nan)
    fig,ax=make_figure(width=15,height=13,left=.25,right=.96,top=.84)
    for i,arm in enumerate(order):
        for j,t in enumerate(targets):
            m=wild[wild.arm.eq(arm)&wild.target.eq(t)]
            if len(m):assert len(m)==1;matrix[i,j]=m.accuracy.iloc[0]
    ax.imshow(np.ma.masked_invalid(matrix),vmin=.3,vmax=1,cmap=rate_cmap(),aspect='auto')
    for i,arm in enumerate(order):
        for j,t in enumerate(targets):
            m=wild[wild.arm.eq(arm)&wild.target.eq(t)]
            text='—' if not len(m) else f'{m.accuracy.iloc[0]:.1%}\n[{m.ci_low.iloc[0]*100:.0f}–{m.ci_high.iloc[0]*100:.0f}]'
            ax.text(j,i,text,ha='center',va='center',fontsize=10,color=heatmap_ink(matrix[i,j],rate_cmap(),.3,1))
    ax.set_xticks(range(4),['User harm','Assistant harm','Either-turn harm','Assistant refusal'])
    ax.set_yticks(range(len(order)),[NAMES[a] for a in order]);ax.tick_params(length=0);ax.grid(False)
    header(ax,'Accuracy changes with the judgment being asked','WildGuardTest · 1,681 pairs; refusal n = 1,680 · cells show accuracy and 95% Wilson interval',
        'Native/BYOC policies are reference views. PolicyLM is supplemental; missing support is a dash.',title_size=20,subtitle_size=12,key_size=11)
    save(fig,out,'03_judgment_accuracy')
    # Operational measurements stay on physical requests; native derivations do not create more calls.
    latency=[]
    for r in bars:
        s=primary[primary.arm.eq(r['arm'])].latency_s.dropna();a,b=bootstrap_mean(s)
        latency.append({**r,'mean':s.mean(),'lo':a,'hi':b,'median':s.median(),'p95':s.quantile(.95)})
    latency.sort(key=lambda r:r['mean']);fig,ax=make_figure(width=14,height=9,left=.22,right=.70,top=.80)
    for i,r in enumerate(latency):
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',color='#2C1FEB',capsize=4,zorder=4)
        ax.scatter(r['p95'],i,marker='|',s=170,color='#956BDD',zorder=3)
        ax.text(1.04,i,f'{r["mean"]:.2f}s mean · {r["median"]:.2f}s median · {r["p95"]:.2f}s p95',transform=ax.get_yaxis_transform(),va='center',fontsize=10)
    ax.set_yticks(range(len(latency)),[NAMES[r['arm']] for r in latency]);ax.invert_yaxis();ax.set_xlabel('Observed final-attempt latency (seconds)')
    clean_axes(ax,despine_left=True,grid_axis='x');header(ax,'Typical latency and the slow tail','Matched Aegis test interaction judgments · concurrent hosted/dedicated runs',
        'Dot: mean with 95% item-bootstrap interval. Tick: p95. Warmup and failed transport attempts excluded.',key_size=10)
    save(fig,out,'04_latency')
    tokens=[]
    for r in bars:
        s=primary[primary.arm.eq(r['arm'])];a,b=bootstrap_mean(s.total_tokens)
        tokens.append({'arm':r['arm'],'input':s.input_tokens.mean(),'output':s.output_tokens.mean(),'lo':a,'hi':b})
    tokens.sort(key=lambda r:r['input']+r['output']);fig,ax=make_figure(width=13,height=9,left=.23,right=.88,top=.80)
    for i,r in enumerate(tokens):
        total=r['input']+r['output'];ax.barh(i,r['input'],color='#956BDD',height=.6);ax.barh(i,r['output'],left=r['input'],color='#2C1FEB',height=.6)
        ax.errorbar(total,i,xerr=[[max(0,total-r['lo'])],[max(0,r['hi']-total)]],fmt='none',ecolor='#18181B',capsize=4,zorder=4)
        ax.text(1.02,i,f'{r["input"]:.0f} in / {r["output"]:.0f} out',transform=ax.get_yaxis_transform(),va='center',fontsize=10)
    ax.set_yticks(range(len(tokens)),[NAMES[r['arm']] for r in tokens]);ax.invert_yaxis();ax.set_xlabel('Mean recorded tokens per judgment')
    clean_axes(ax,despine_left=True,grid_axis='x');header(ax,'Token use includes the decision interface','Aegis test · interaction harm · full original pairs',
        'Purple: input; indigo: output. 95% bootstrap total-token intervals. Tokenizers differ across providers.',key_size=10.5)
    save(fig,out,'05_tokens')
    arms=metric.arm.tolist();fig,ax=make_figure(width=14,height=12,left=.23,right=.96,top=.85)
    values=np.full((len(arms),len(arms)),np.nan);cells={}
    sub=pairs[pairs.source_dataset.eq('aegis_v2_test') & pairs.input_unit.eq('conversation') & pairs.target.eq('interaction_harm')]
    for i,a in enumerate(arms):
        for j,b in enumerate(arms):
            if a==b:values[i,j]=0;continue
            p=sub[((sub.arm_a==a)&(sub.arm_b==b))|((sub.arm_a==b)&(sub.arm_b==a))].iloc[0]
            values[i,j]=p.disagreement;cells[(i,j)]=p
    ax.imshow(values,vmin=0,vmax=max(.2,np.nanmax(values)),cmap=sequential_cmap(),aspect='auto')
    for i in range(len(arms)):
        for j in range(len(arms)):
            p=cells.get((i,j));text='—' if p is None else f'{p.disagreement:.0%}\n[{p.ci_low*100:.0f}–{p.ci_high*100:.0f}]'
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color=heatmap_ink(values[i,j],sequential_cmap(),0,max(.2,np.nanmax(values))))
    ax.set_xticks(range(len(arms)),range(1,len(arms)+1));ax.set_yticks(range(len(arms)),[f'{i+1}. {NAMES[a]}' for i,a in enumerate(arms)])
    ax.tick_params(length=0);ax.grid(False);header(ax,'Models can disagree even at similar accuracy','Aegis test interaction harm · paired, jointly valid outputs only',
        'Cells: disagreement and 95% Wilson interval. Column numbers match rows; counts are in pairwise_disagreement.csv.',subtitle_size=13,key_size=10.5)
    save(fig,out,'06_model_disagreement')
    trade=[]
    for r in bars:trade.append({**r,'accuracy':r['unsafe_recall'],'ci_low':r['unsafe_recall_low'],'ci_high':r['unsafe_recall_high'],
        'x':r['safe_false_unsafe_rate'],'xlo':r['safe_false_unsafe_rate_low'],'xhi':r['safe_false_unsafe_rate_high']})
    scatter(trade,out,'07_harm_tradeoff','Finding unsafe content versus false alarms','Aegis test · unsafe recall against human-safe pairs labeled unsafe',
        '95% Wilson intervals; failed outputs count as missed unsafe cases, not false-unsafe predictions.', 'Human-safe pairs labeled unsafe (fraction)',ylabel='Unsafe recall')
    dedicated=q[q.source_dataset.eq('aegis_v2_test') & q.input_unit.eq('conversation') & q.target.eq('interaction_harm') & q.provider.eq('baseten')]
    proxy=[]
    for r in dedicated.to_dict('records'):
        g=df[df.arm.eq(r['arm']) & df.source_dataset.eq('aegis_v2_test') & df.input_unit.eq('conversation') & df.target.eq('interaction_harm')]
        proxy.append({**r,'x':g.gpu_allocated_proxy_usd.mean()*1000})
    scatter(sorted(proxy,key=lambda r:r['x']),out,'08_dedicated_cost_scenario','Dedicated GPU cost allocation versus accuracy','Aegis test · runner-time allocation scenario; NOT an invoice or marginal production cost',
        'Includes recorded startup/idle; Qwen shared by request-time fraction. Native/BYOC points use parent-call costs.',
        'Allocated GPU scenario per 1,000 judgments (USD)')
    # Keep source/input-unit variation visible instead of pooling it away.
    cohorts=[('aegis_v2_validation','conversation','interaction_harm','Aegis validation\npaired · n=486'),
             ('aegis_v2_test','conversation','interaction_harm','Aegis test\npaired · n=813'),
             ('wildguardtest','conversation','interaction_harm','WildGuardTest\npaired · n=1,681'),
             ('aegis_v2_validation','prompt_only','interaction_harm','Aegis validation\nprompt-only · n=709'),
             ('aegis_v2_test','prompt_only','interaction_harm','Aegis test\nprompt-only · n=1,113'),
             ('xstest','prompt_only','prompt_harm','XSTest\nprompt-only · n=450')]
    fig,ax=make_figure(width=17,height=11,left=.20,right=.97,top=.81,bottom=.12)
    matrix=np.full((len(arms),len(cohorts)),np.nan);annotations={}
    for i,arm in enumerate(arms):
        for j,(source,unit,target,_) in enumerate(cohorts):
            m=q[q.arm.eq(arm)&q.source_dataset.eq(source)&q.input_unit.eq(unit)&q.target.eq(target)]
            if len(m):
                assert len(m)==1;r=m.iloc[0];matrix[i,j]=r.accuracy
                annotations[(i,j)]=f'{r.accuracy:.1%}\n[{r.ci_low*100:.0f}–{r.ci_high*100:.0f}]'
    ax.imshow(np.ma.masked_invalid(matrix),vmin=.3,vmax=1,cmap=rate_cmap(),aspect='auto')
    for i in range(len(arms)):
        for j in range(len(cohorts)):
            ax.text(j,i,annotations.get((i,j),'—'),ha='center',va='center',fontsize=10,
                    color=heatmap_ink(matrix[i,j],rate_cmap(),.3,1))
    ax.set_xticks(range(len(cohorts)),[c[3] for c in cohorts],fontsize=10);ax.set_yticks(range(len(arms)),[NAMES[a] for a in arms]);ax.grid(False);ax.tick_params(length=0)
    header(ax,'Safety agreement varies across datasets and input units','Common-policy interfaces · full paired inputs or original prompt-only inputs, kept separate',
        'Cells: strict accuracy with 95% Wilson interval. XSTest asks prompt harm; other columns ask interaction harm.',subtitle_size=13,key_size=11)
    save(fig,out,'09_dataset_variation')
    return primary,metric


def markdown_table(frame,columns,formats=None):
    formats=formats or {};lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in frame.to_dict('records'):
        vals=[]
        for c in columns:
            v=r.get(c);vals.append('—' if pd.isna(v) else formats[c](v) if c in formats else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def report(df,q,summary,primary,metric,rates,allocations,out):
    table=[]
    for r in metric.to_dict('records'):
        s=primary[primary.arm.eq(r['arm'])]
        table.append({'Model':NAMES[r['arm']],'Accuracy':f'{r["accuracy"]:.1%}','95% CI':f'{r["ci_low"]:.1%}–{r["ci_high"]:.1%}',
                      'Unsafe recall':f'{r["unsafe_recall"]:.1%}','Safe false alarm':f'{r["safe_false_unsafe_rate"]:.1%}',
                      'Median seconds':f'{s.latency_s.median():.2f}',
                      'Token $ / 1k':f'{s.token_cost_usd.mean()*1000:.3f}' if s.token_cost_usd.notna().all() else 'GPU billing',
                      'Mean in / out':f'{s.input_tokens.mean():.0f} / {s.output_tokens.mean():.0f}'})
    total=summary.estimated_token_cost_usd.sum();old=summary.historical_estimated_token_cost_usd.sum()
    text=f'''# Full source safety evaluations — descriptive report

All three source runs are verified complete: **124,773 physical predictions and 14,900 reused derived judgments**. Headline metric: `source_accuracy`, strict agreement with published human labels (or the explicitly declared either-turn OR). Model refusals, invalid outputs, and truncations count as incorrect. This is source-label agreement, not an independently adjudicated policy audit or a new held-out/preregistered result; prior development exposure remains disclosed.

## Matched safety comparison

Aegis test, original prompt-response pairs, interaction harm: **813 identical examples per common-policy interface**. Native/BYOC guards appear separately in the judgment and dedicated-cost views. These intervals describe item variability on this source, not training-run uncertainty; models are paired, so overlapping intervals are not a significance test.

{markdown_table(pd.DataFrame(table),list(table[0]))}

## Charts

'''
    titles=['Hosted cost versus accuracy','Matched safety accuracy','Judgment-type accuracy','Latency and slow tails','Input/output token use','Pairwise model disagreement','Unsafe recall versus false alarms','Dedicated GPU allocation scenario','Variation across datasets and input units']
    files=sorted(out.glob('*.png'))
    for title,f in zip(titles,files):text+=f'### {title}\n\n![{title}]({f.name})\n\n'
    text+=f'''## Descriptive statistics and denominators

- `scorer_stats.csv`: every observed scorer for every model × dataset × input unit × target × scheme × rubric × policy mode × physical/derived configuration. Includes count, missing count, coverage, mean, sample standard deviation, min/max, p05/p25/median/p75/p95/p99, sum, and mean interval. Zero-coverage scorers remain blank, never zero-filled.
- `resource_stats.csv`: the same summaries for cost, latency, input/output/total/cache tokens, PolicyLM characters/windows, and GPU allocation proxy. Derived rows have no physical latency/tokens; their `parent_inference_time_s` and allocated path cost reuse parents. Do not add derived resource totals to physical-call totals.
- `quality_by_configuration.csv`: strict accuracy, unsafe recall, safe false-unsafe rate, refusal recall, and safe refusal false-positive rate with explicit class denominators and Wilson intervals.
- `pairwise_disagreement.csv`: paired label disagreement across model interfaces within each source/input/target. Denominator is jointly valid output pairs; failure rates are reported separately. Cross-policy comparisons are flagged.
- `operational_by_arm.csv`: physical requests only. Overall task mixes differ; this is an operational inventory, not a pooled quality ranking.
- `resource_stats_by_arm.csv`: full physical-call resource distributions per arm, across its actual workload mix.
- `scorer_stats_by_arm_target.csv`: aggregate scorer distributions per model/target/configuration across sources; use the fully stratified table for comparisons.
- `scorer_definitions.csv`: meaning, denominator, and direction for every configured scorer, including the inapplicable lookalike scorer.
- `judgments.csv.gz`: tidy, reproducible row-level scores, labels, model revisions, run IDs, resource observations, and source IDs. No raw prompts/responses or credentials are exported.

In `quality_by_configuration.csv`, unsafe recall uses ALL human-unsafe cases (failures count as misses), and `safe_false_unsafe_rate` uses all human-safe cases. Derived `refusal_recall` and `refusal_false_positive_rate` use human refusal/compliance labels. The original recorded scorer `unsafe_recall` excludes failed/caution predictions, and original `refusal_recall` excludes failed predictions, so their means can differ from the strict class rates. The original `safe_refusal_fp` means a human-safe REFUSING response labeled UNSAFE in the response-harm task; it is not false-positive refusal detection. Scorer definitions and non-missing denominators are explicit in CSV. Averaging unrelated scorer dimensions is not accuracy. The three `binary_correct_nc_*` variants use alternate caution treatment and exclude invalid predictions; native caution counts as wrong under strict source agreement. Probability metrics exist only where probabilities were actually returned. `lookalike_fp` has no eligible stratum in these final source runs and is exported as missing, not zero.

## Cost and measurement scope

Current-rate recorded hosted token estimate: **${total:.2f}**. The unchanged October 9 rate snapshot gives **${old:.2f}** on the same final predictions. These exclude subscriptions, gateway fees, unrecorded transport retries, regional/long-context premiums, and credits; they are estimates, not invoices. Public rates were rechecked October 12. Clef Flash is now published at $0.038/M input versus $0.09/M in the historical snapshot. [Cloudflare Clef Flash pricing]({rates['clef_flash_choice']['source']}). Other primary references: [TypeSafe]({rates['jev_choice']['source']}), [Clef]({rates['clef_choice']['source']}), [OpenAI standard pricing]({rates['luna_structured']['source']}), [Decisions pricing]({rates['decisions_luna']['source']}), [Claude]({rates['claude_structured']['source']}). The historical snapshot is preserved.

Dedicated Baseten models have recorded token counts but no per-token invoice in this export. GPU scenario totals use **{allocations.recorded_runner_minutes_proxy.iloc[0]:.2f} recorded runner minutes** times the selected two-L4, one-L4, or one-H100 resources; total allocation scenario **${allocations.gpu_proxy_total_usd.sum():.2f}**. This runtime proxy can include inactive replay time and omit startup outside the runner, so it is neither measured billed spend nor a strict bound. Shared Qwen resources are allocated by the two interfaces' cumulative recorded request latency; this is an explicit accounting choice, not measured GPU utilization. Costs per judgment depend on concurrency, throughput, idle time, and workload mix. Native OR and Nemotron component views reuse parent-call costs without additional inference. [Baseten resource pricing]({rates['baseten_dedicated']['source']}). PolicyLM's local CPU/compute spend and token counts are unavailable; absent values are not free usage.

Latency is the recorded final-attempt request duration, from concurrent runs with restarts and network interruptions. API warmup and failed transport attempts are excluded from successful-final-attempt latency; this is not a controlled serving-speed benchmark. Percentile intervals in figures describe item variability. Continuous-score mean intervals in CSV use normal standard-error intervals; binary scorer intervals use Wilson. Tokenizers, adapter instructions, and wrappers differ across providers.

## Coverage and native-policy caveats

Aegis validation: 486 paired + 709 prompt-only; Aegis test: 813 paired + 1,113 prompt-only; WildGuardTest: 1,681 paired; XSTest: 450 prompt-only. Refusal evaluation excludes one WildGuard example without its source refusal label (n = 1,680). Aegis does not publish independent turn-harm labels; native turn views there stay unscored. WildGuard publishes both turn labels and refusal labels; either-turn harm is the study's declared OR. Native/BYOC rules and ternary caution may differ from common-policy instructions. PolicyLM is prompt-only and supplemental because of its published training/development-overlap caveat.

Recreate with `.venv/bin/python -m analysis.source_report`. All inputs are the audited final checkpoints; pilots and legacy runs are excluded.
'''
    (out/'report.md').write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT);args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True);use_braintrust_theme()
    df,rates,receipts,hashes,allocations=load();scores,resources,summary=descriptive_tables(df);q=quality(df);pairs=disagreement(df)
    for name,frame in [('scorer_stats',scores),('resource_stats',resources),('operational_by_arm',summary),('quality_by_configuration',q),('pairwise_disagreement',pairs),('gpu_cost_allocation_scenario',allocations)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    overall=[];by_target=[]
    for arm,g in df[df.record_kind.eq('physical')].groupby('arm'):
        for col in RESOURCES:overall.append({'arm':arm,'display_name':NAMES[arm],'metric':col,**describe(g[col],len(g),binary=False)})
    target_keys=['arm','target','scheme','rubric','policy_mode','record_kind']
    for key,g in df.groupby(target_keys):
        for name in SCORERS:
            by_target.append({**dict(zip(target_keys,key)),'scorer':name,**describe(g['score__'+name],len(g),binary=name not in {'p_unsafe','brier_unsafe','top_margin'})})
    pd.DataFrame(overall).to_csv(out/'resource_stats_by_arm.csv',index=False)
    pd.DataFrame(by_target).to_csv(out/'scorer_stats_by_arm_target.csv',index=False)
    pd.DataFrame([{'scorer':k,'meaning':v[0],'denominator':v[1],'direction':v[2]} for k,v in SCORERS.items()]).to_csv(out/'scorer_definitions.csv',index=False)
    df.to_csv(out/'judgments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    primary,metric=plots(df,q,pairs,out);report(df,q,summary,primary,metric,rates,allocations,out)
    metadata={'physical_predictions':int(df.physical_calls.sum()),'derived_judgments':int(df.record_kind.eq('derived').sum()),
        'configurations':len(q),'scorers':sorted(c[7:] for c in df if c.startswith('score__')),
        'source_hashes':hashes,'verification_receipts':receipts,'pricing':rates,
        'recorded_hosted_token_estimate_usd':float(summary.estimated_token_cost_usd.sum()),
        'gpu_cost_is_allocation_scenario_not_invoice':True}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(out/'safety-analysis.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name!='safety-analysis.zip':archive.write(p,p.name)
    print(json.dumps({'output':str(out),'physical_predictions':metadata['physical_predictions'],'scored_configurations':len(q),
        'scorers':metadata['scorers'],'hosted_estimate_usd':metadata['recorded_hosted_token_estimate_usd']}))


if __name__=='__main__':main()
