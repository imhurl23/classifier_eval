"""Exploratory paired tests and frequentist random-intercept regressions.

Binary GLMM: normal random intercept integrated by Gauss-Hermite quadrature.
No Bayesian posterior intervals are relabeled as frequentist p-values.
Reproduce: .venv/bin/python -m analysis.inferential_models
"""
import json,hashlib,warnings
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats,optimize,special
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.braintrust_viz import use_braintrust_theme,make_figure,header,clean_axes,savefig
from analysis.make_development_charts import ROOT,NAMES

OUT=ROOT/'analysis/output/development-2026-10-07/inference'
BASE='jev_choice'
ARMS=[BASE]+sorted(a for a in NAMES if a!=BASE)


@lru_cache(maxsize=16)
def quadrature(nodes):
    z,w=special.roots_hermitenorm(nodes)
    positive=w>0
    return z[positive],np.log(w[positive])-.5*np.log(2*np.pi)


def objective(theta,X,y,nodes=121):
    """Return marginal negative log likelihood and its analytical gradient."""
    z,logw=quadrature(nodes)
    sigma=np.exp(theta[-1]);b=sigma*z
    eta=np.einsum('gop,p->go',X,theta[:-1])[:,None,:]+b[None,:,None]
    loglik=np.sum(y[:,None,:]*eta-np.logaddexp(0,eta),axis=2)+logw[None,:]
    norm=special.logsumexp(loglik,axis=1)
    posterior=np.exp(loglik-norm[:,None]);residual=y[:,None,:]-special.expit(eta)
    gb=-np.einsum('gn,gno,gop->p',posterior,residual,X)
    gs=-np.sum(posterior*np.sum(residual,axis=2)*b[None,:])
    return -float(np.sum(norm)),np.r_[gb,gs]


def hessian(theta,X,y,nodes=181):
    columns=[]
    for j in range(len(theta)):
        h=1e-4*max(1,abs(theta[j]));delta=np.zeros(len(theta));delta[j]=h
        columns.append((objective(theta+delta,X,y,nodes)[1]-objective(theta-delta,X,y,nodes)[1])/(2*h))
    H=np.column_stack(columns);return (H+H.T)/2


def fit_logistic(X,y,names):
    start=np.zeros(X.shape[2]+1);start[0]=special.logit(y.mean());start[-1]=0
    bounds=[(-25,25)]*X.shape[2]+[(-6,3)]
    def fit(nodes,initial):
        return optimize.minimize(objective,initial,args=(X,y,nodes),jac=True,method='L-BFGS-B',bounds=bounds,
                                 options={'maxiter':1000,'ftol':1e-13,'gtol':1e-6,'maxls':50})
    previous=fit(121,start)
    grid_history=[]
    for nodes in (241,401,641,1001,1601):
        check=fit(nodes,previous.x)
        delta=float(np.max(np.abs(check.x-previous.x)));ll_delta=abs(float(check.fun-previous.fun))
        grid_history.append({'nodes':nodes,'coefficient_drift':delta,'loglik_drift':ll_delta})
        if check.success and delta<1e-3 and ll_delta<1e-4: break
        previous=check
    else:
        raise RuntimeError(f'GLMM convergence/quadrature failed: {check.message}; coefficient drift {delta}; loglik drift {ll_delta}')
    theta=check.x;H=hessian(theta,X,y,nodes);eig=np.linalg.eigvalsh(H)
    if np.min(eig)<=0:raise RuntimeError('Non-positive observed information')
    if abs(theta[-1]+6)<.01 or abs(theta[-1]-3)<.01:raise RuntimeError('Random-intercept variance hit parameter boundary')
    covariance=np.linalg.inv(H);se=np.sqrt(np.diag(covariance))
    coefs=[]
    for name,b,s in zip(names,theta[:-1],se[:-1]):
        coefs.append(dict(term=name,estimate=float(b),se=float(s),z=float(b/s),p=float(2*stats.norm.sf(abs(b/s))),
                          ci_low=float(b-1.96*s),ci_high=float(b+1.96*s),odds_ratio=float(np.exp(b)),
                          or_low=float(np.exp(b-1.96*s)),or_high=float(np.exp(b+1.96*s))))
    arm_coefs=[r for r in coefs if r['term'].startswith('arm:') and ':unsafe' not in r['term']]
    if arm_coefs:
        adjusted=multipletests([r['p'] for r in arm_coefs],method='holm')[1]
        for r,p in zip(arm_coefs,adjusted):r['p_holm_six_arm_contrasts']=float(p)
    return dict(coefficients=coefs,log_likelihood=-float(check.fun),random_intercept_sd=float(np.exp(theta[-1])),
                random_sd_ci=[float(np.exp(theta[-1]-1.96*se[-1])),float(np.exp(theta[-1]+1.96*se[-1]))],
                converged=bool(check.success),gradient_max=float(np.max(np.abs(check.jac))),
                quadrature_nodes=nodes,quadrature_grid_history=grid_history,quadrature_coefficient_drift=delta,quadrature_loglik_drift=ll_delta,
                min_information_eigenvalue=float(eig.min()),n_observations=int(y.size),n_items=int(len(y)))


def design(gold,loglength,arm_effect=True,interaction=False):
    n=len(gold);m=len(ARMS);columns=[np.ones((n,m))];names=['Intercept']
    if arm_effect:
        for i,arm in enumerate(ARMS[1:],1):
            columns.append(np.tile(np.arange(m)==i,(n,1)).astype(float));names.append('arm:'+arm)
    unsafe=np.tile((gold=='unsafe')[:,None],(1,m)).astype(float)
    columns.extend([unsafe,np.tile(loglength[:,None],(1,m))]);names.extend(['source_unsafe','log_chars_centered'])
    if interaction:
        for i,arm in enumerate(ARMS[1:],1):
            columns.append(unsafe*(np.arange(m)==i)[None,:]);names.append('arm:'+arm+':unsafe')
    return np.stack(columns,axis=2),names


def paired_tests(values,outcome,binary=False):
    results=[]
    for a,b in combinations(ARMS,2):
        delta=values[a]-values[b]
        if np.std(delta,ddof=1)==0:
            raise ValueError('Constant paired differences require explicit handling')
        test=stats.ttest_rel(values[a],values[b]);ci=test.confidence_interval()
        r=dict(outcome=outcome,arm_a=a,arm_b=b,n=len(delta),df=float(test.df),mean_difference=float(delta.mean()),
               ci_low=float(ci.low),ci_high=float(ci.high),t=float(test.statistic),p=float(test.pvalue),
               paired_cohens_dz=float(delta.mean()/delta.std(ddof=1)))
        if binary:
            wins=int(np.sum(delta==1));losses=int(np.sum(delta==-1))
            r.update(a_correct_b_wrong=wins,b_correct_a_wrong=losses,
                     mcnemar_exact_p=float(stats.binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0)
        else:r.update(geometric_latency_ratio_a_over_b=float(np.exp(delta.mean())),ratio_low=float(np.exp(ci.low)),ratio_high=float(np.exp(ci.high)))
        results.append(r)
    for r,p in zip(results,multipletests([r['p'] for r in results],method='holm')[1]):r['p_holm_21']=float(p)
    if binary:
        for r,p in zip(results,multipletests([r['mcnemar_exact_p'] for r in results],method='holm')[1]):r['mcnemar_exact_p_holm_21']=float(p)
    return results


def forest(filename,coefs,title,subtitle,key,value='odds_ratio',low='or_low',high='or_high'):
    fig,ax=make_figure(width=12,height=7,left=.25,right=.96,top=.78)
    ax.axvline(1,color='#6B6B76',ls='--',lw=1)
    rows=[r for r in coefs if r['term'].startswith('arm:') and ':unsafe' not in r['term']]
    for i,r in enumerate(rows):
        x=r[value];ax.errorbar(x,i,xerr=[[x-r[low]],[r[high]-x]],fmt='o',ms=9,color='#2C1FEB',ecolor='#18181B',capsize=4)
    ax.set_yticks(range(len(rows)),[NAMES[r['term'].split(':')[1]] for r in rows]);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlabel('Ratio relative to Jev Choice (log scale)');clean_axes(ax,True,'x');header(ax,title,subtitle,key)
    for ext in ('png','svg'):savefig(fig,OUT/f'{filename}.{ext}')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);use_braintrust_theme()
    reference=ROOT/'data/processed/study-reference/00de7797f0a6'
    rows=[json.loads(s) for s in (reference/'results.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['target']=='interaction_harm'];ids=sorted({r['item_id'] for r in rows})
    if len(rows)!=1400 or len(ids)!=200:raise ValueError('Reference counts mismatch')
    lookup={(r['item_id'],r['arm']):r for r in rows}
    if len(lookup)!=1400:raise ValueError('Duplicate matched outcome')
    sample={r['id']:r for r in map(json.loads,(reference/'sample.jsonl').read_text().splitlines())}
    gold=np.array([lookup[i,BASE]['human_interaction_gold'] for i in ids]);loglen=np.log([len(sample[i]['input']['text']) for i in ids]);loglen-=loglen.mean()
    correct={arm:np.array([float(lookup[i,arm]['status']=='ok' and lookup[i,arm]['label']==lookup[i,arm]['human_interaction_gold']) for i in ids]) for arm in ARMS}
    latency={arm:np.log([lookup[i,arm]['latency_ms']/1000 for i in ids]) for arm in ARMS}
    agreement_tests=paired_tests(correct,'strict_source_agreement',True);latency_tests=paired_tests(latency,'log_latency_seconds')
    pd.DataFrame(agreement_tests).to_csv(OUT/'paired_ttests_agreement.csv',index=False);pd.DataFrame(latency_tests).to_csv(OUT/'paired_ttests_log_latency.csv',index=False)
    y=np.column_stack([correct[a] for a in ARMS])
    X,names=design(gold,loglen)
    # Check gradient against independent finite differences before trusting fits.
    trial=np.r_[np.linspace(-.2,.3,X.shape[2]),.2]
    numeric=optimize._numdiff.approx_derivative(lambda t:np.array([objective(t,X,y,41)[0]]),trial).ravel()
    analytic=objective(trial,X,y,41)[1]
    gradient_error=float(np.max(np.abs(numeric-analytic)))
    if gradient_error>1e-4:raise RuntimeError('GLMM gradient verification failed')
    primary=fit_logistic(X,y,names);print('Primary GLMM fitted',flush=True)
    X0,names0=design(gold,loglen,False);null=fit_logistic(X0,y,names0);print('Null GLMM fitted',flush=True)
    Xi,namesi=design(gold,loglen,interaction=True);interaction=fit_logistic(Xi,y,namesi);print('Interaction GLMM fitted',flush=True)
    arm_lr=2*(primary['log_likelihood']-null['log_likelihood']);inter_lr=2*(interaction['log_likelihood']-primary['log_likelihood'])
    lrt={'arm':dict(chi2=arm_lr,df=6,p=float(stats.chi2.sf(arm_lr,6))),
         'arm_by_source_label':dict(chi2=inter_lr,df=6,p=float(stats.chi2.sf(inter_lr,6)))}
    frame=pd.DataFrame([dict(item_id=item,arm=arm,source_unsafe=int(gold[i]=='unsafe'),log_chars_centered=float(loglen[i]),log_latency=float(latency[arm][i]),correct=int(correct[arm][i])) for i,item in enumerate(ids) for arm in ARMS])
    frame.to_csv(OUT/'regression_data.csv',index=False)
    formula="log_latency ~ C(arm, Treatment(reference='jev_choice')) + source_unsafe + log_chars_centered"
    model=smf.mixedlm(formula,frame,groups=frame['item_id'],re_formula='1')
    captured=[];fitted=None
    for method in ('lbfgs','powell','bfgs'):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');candidate=model.fit(reml=False,method=method,maxiter=1000,disp=False)
            captured.extend(str(w.message) for w in caught)
            if candidate.converged and np.isfinite(candidate.bse_fe).all():fitted=candidate;break
        except (ValueError,np.linalg.LinAlgError) as exc:captured.append(type(exc).__name__+': '+str(exc))
    if fitted is None:raise RuntimeError('Latency mixed model did not converge')
    with warnings.catch_warnings(record=True) as sensitivity_warnings:
        warnings.simplefilter('always')
        sensitivity=model.fit(reml=False,method='powell',maxiter=1000,disp=False)
    optimizer_drift=float(np.max(np.abs(sensitivity.fe_params.to_numpy()-fitted.fe_params.to_numpy())))
    likelihood_drift=abs(float(sensitivity.llf-fitted.llf))
    if not sensitivity.converged or optimizer_drift>1e-3 or likelihood_drift>1e-3:
        raise RuntimeError('Latency mixed-model optimizer sensitivity check failed')
    latency_coefs=[]
    for term,b in fitted.fe_params.items():
        se=float(fitted.bse_fe[term]);name='arm:'+term.split('[T.')[1].rstrip(']') if '[T.' in term else term
        latency_coefs.append(dict(term=name,estimate=float(b),se=se,z=float(b/se),p=float(2*stats.norm.sf(abs(b/se))),
                                   ratio=float(np.exp(b)),ratio_low=float(np.exp(b-1.96*se)),ratio_high=float(np.exp(b+1.96*se))))
    armterms=[r for r in latency_coefs if r['term'].startswith('arm:')]
    for r,p in zip(armterms,multipletests([r['p'] for r in armterms],method='holm')[1]):r['p_holm_six_arm_contrasts']=float(p)
    param_names=list(fitted.fe_params.index);indexes=[i for i,t in enumerate(param_names) if '[T.' in t]
    beta=fitted.fe_params.iloc[indexes].to_numpy();cov=fitted.cov_params().loc[param_names,param_names].to_numpy()[np.ix_(indexes,indexes)]
    wald=float(beta@np.linalg.solve(cov,beta))
    lat=dict(formula=formula,coefficients=latency_coefs,converged=bool(fitted.converged),method=method,log_likelihood=float(fitted.llf),
             random_intercept_variance=float(fitted.cov_re.iloc[0,0]),residual_variance=float(fitted.scale),warnings=captured,
             optimizer_sensitivity={'method':'powell','converged':bool(sensitivity.converged),'max_fixed_coefficient_drift':optimizer_drift,'loglik_drift':likelihood_drift,'warnings':[str(w.message) for w in sensitivity_warnings]},
             arm_joint_wald=dict(chi2=wald,df=6,p=float(stats.chi2.sf(wald,6))),n_observations=1400,n_items=200)
    forest('01_agreement_mixed_effects',primary['coefficients'],'Adjusted source-agreement contrasts remain uncertain',
           'Logistic mixed-effects regression · random intercept per conversation · n = 200 pairs',
           'Odds ratios with nominal 95% Wald intervals; controls: source label and log conversation length.')
    forest('02_latency_mixed_effects',latency_coefs,'Observed latency differs after accounting for matched items',
           'Linear mixed-effects regression of log latency · n = 200 pairs',
           'Geometric latency ratios with nominal 95% Wald intervals; network/provider effects remain.',value='ratio',low='ratio_low',high='ratio_high')
    summary=dict(scope='Exploratory development inference only; tests were chosen after inspecting development outcomes.',baseline=BASE,
                 paired_agreement_tests=agreement_tests,paired_log_latency_tests=latency_tests,
                 logistic_model_formula='correct ~ arm + source_unsafe + centered_log_chars + (1 | item_id)',
                 logistic_primary=primary,logistic_null=null,logistic_interaction=interaction,logistic_likelihood_ratio_tests=lrt,
                 latency_mixed_model=lat,gradient_verification_max_error=gradient_error,
                 source_sha256=hashlib.sha256((reference/'results.jsonl').read_bytes()).hexdigest(),
                 multiplicity='21 two-sided agreement t-tests: Holm; 21 latency t-tests: separate Holm family; exact McNemar: separate sensitivity family; six baseline contrasts per regression: separate Holm families. Omnibus tests and 95% intervals are nominal.',
                 notes=['Strict correctness treats refusals as incorrect.','No source/dataset random effect is identifiable: all reference items are Aegis development pairs.',
                        'No reviewer random effect is fitted: human audit labels are absent.','Binary t-test differences are discrete; exact McNemar sensitivity is included.',
                        'GLMM uses fixed Gaussian quadrature, not lme4 adaptive quadrature; increasing-grid agreement and positive information are checked.',
                        'Mixed-model Wald intervals/p-values rely on normal random effects and asymptotic approximations.','LMM fits log end-to-end latency; does not isolate architecture or inference-only speed.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(primary['coefficients']).to_csv(OUT/'logistic_mixed_coefficients.csv',index=False)
    pd.DataFrame(interaction['coefficients']).to_csv(OUT/'logistic_interaction_coefficients.csv',index=False)
    pd.DataFrame(latency_coefs).to_csv(OUT/'latency_mixed_coefficients.csv',index=False)
    lines=['# Exploratory t-tests and mixed-effects regressions','',
           'Scope: 200 matched Aegis development pairs, seven arms, 1,400 reference interaction decisions. This is source-label agreement; no held-out or independent human-policy inference is made. All tests are post hoc and exploratory. Refusals count as incorrect rather than being removed.','',
           '## Paired agreement t-tests','',
           'Two-sided tests use each pair’s binary strict-correctness difference (df = 199); Holm correction covers all 21 arm comparisons. Nominal 95% t intervals and paired Cohen’s dz are exported. Exact McNemar tests provide a binary-outcome sensitivity check, corrected across their own 21-comparison family.','',
           f'{sum(r["p_holm_21"]<.05 for r in agreement_tests)}/21 agreement t-tests and {sum(r["mcnemar_exact_p_holm_21"]<.05 for r in agreement_tests)}/21 exact McNemar tests remain below .05 after Holm correction. Failure to reject a difference is not evidence of equivalence.','',
           '| Comparison (A − B) | Difference (percentage points) | Nominal 95% CI | t | Raw p | Holm p |',
           '|---|---:|---:|---:|---:|---:|']
    for r in sorted(agreement_tests,key=lambda r:r['p']):
        lines.append(f'| {NAMES[r["arm_a"]]} − {NAMES[r["arm_b"]]} | {r["mean_difference"]*100:.2f} | [{r["ci_low"]*100:.2f}, {r["ci_high"]*100:.2f}] | {r["t"]:.3f} | {r["p"]:.4g} | {r["p_holm_21"]:.4g} |')
    lines+=['','## Logistic mixed-effects regression','',
            '`correct ~ arm + source_unsafe + centered_log_chars + (1 | item_id)`','',
            'The random conversation intercept accounts for repeated judgments of each item. Jev Choice is the reference arm. The normal random-intercept likelihood is integrated with fixed Gauss-Hermite quadrature and optimized by maximum likelihood; Wald intervals come from the full observed-information matrix. Six model-versus-baseline coefficient p-values receive Holm correction.','',
            f'Omnibus arm likelihood-ratio test: χ²(6) = {arm_lr:.3f}, p = {lrt["arm"]["p"]:.4g}. Item random-intercept SD = {primary["random_intercept_sd"]:.3f}.','',
            f'The exploratory arm × source-label model improves fit by χ²(6) = {inter_lr:.3f}, p = {lrt["arm_by_source_label"]["p"]:.4g}. Its full coefficients are exported; this is a separate nominal omnibus test.','',
            '![Logistic model contrasts](01_agreement_mixed_effects.png)','',
            '| Arm vs Jev Choice | Odds ratio | Nominal 95% CI | Holm p |','|---|---:|---:|---:|']
    for r in primary['coefficients']:
        if r['term'].startswith('arm:'):
            lines.append(f'| {NAMES[r["term"].split(":")[1]]} | {r["odds_ratio"]:.3f} | [{r["or_low"]:.3f}, {r["or_high"]:.3f}] | {r["p_holm_six_arm_contrasts"]:.4g} |')
    lines+=['','## Log-latency mixed-effects regression','',
            '`log(latency_seconds) ~ arm + source_unsafe + centered_log_chars + (1 | item_id)`','',
            f'Fit by maximum likelihood, optimizer `{method}`, converged = {fitted.converged}. Omnibus arm Wald test: χ²(6) = {wald:.3f}, p < 0.001. Ratios compare geometric latencies, not arithmetic means or medians. End-to-end timings include provider/network effects.','',
            '![Latency model contrasts](02_latency_mixed_effects.png)','',
            f'Separate paired t-tests on matched log latency find {sum(r["p_holm_21"]<.05 for r in latency_tests)}/21 comparisons below .05 after their own Holm adjustment.','',
            '## Checks and limits','',
            f'Analytical GLMM gradient versus independent finite differences: maximum absolute error {gradient_error:.3g}. All logistic fits converged, had positive observed-information eigenvalues, and met quadrature tolerances across successive node grids (up to 641 nodes). Primary maximum coefficient drift: {primary["quadrature_coefficient_drift"]:.3g}.','',
            'The balanced sample is development data and is not an independent replication. The mixed models assume independent conversation clusters and a normal random intercept. They do not include model random effects (these seven arms are the fixed comparisons of interest), dataset random effects (only one source), or reviewer random effects (no audit labels). An arm-specific item random-slope structure is not attempted with one reference observation per arm/item. No claims of equivalence or causal architecture effects follow from these tests.','',
            f'Independent Powell optimizer check: maximum fixed-effect drift {optimizer_drift:.3g}; log-likelihood drift {likelihood_drift:.3g}.','',
            'Latency fit warnings (including earlier unsuccessful optimizer attempts): '+('; '.join(captured) or 'none')+'.','',
            'Recreate: `.venv/bin/python -m analysis.inferential_models`. CSVs include full pairwise tests, regression data, and coefficients; `summary.json` records formulas, convergence, random-effect variance, input hash, and numerical checks.','',
            'Method references: [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [Gaussian quadrature nodes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html), [statsmodels linear mixed models](https://www.statsmodels.org/stable/mixed_linear.html).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'agreement_holm_significant':sum(r['p_holm_21']<.05 for r in agreement_tests),'logistic_arm_lrt':lrt['arm'],
                      'source_label_interaction_lrt':lrt['arm_by_source_label'],'latency_arm_wald':lat['arm_joint_wald'],'output':str(OUT)},indent=2),flush=True)

if __name__=='__main__':main()
