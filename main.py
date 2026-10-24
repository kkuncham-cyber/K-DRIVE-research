"""K-DRIVE synthetic enterprise digital-readiness proof of concept.

Original simulation, knowledge rule and evaluation implementation.
Run: python main.py  (Python 3.11+, numpy, pandas, sklearn, scipy, matplotlib).
All rows are simulated: never describe the results as firm-level field validation.
"""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score,
                             f1_score, precision_score, recall_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261024
OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(exist_ok=True)
FEATURES = ['data_maturity', 'integration', 'digital_skills', 'leadership',
            'innovation', 'entrepreneurial_agility', 'cyber_governance',
            'investment_capacity']
CATEGORICAL = ['sector', 'firm_size']
PREDICTORS = FEATURES + CATEGORICAL
POLICIES = ['Score threshold', 'Cost-aware (no rules)', 'Knowledge rules only', 'K-DRIVE']


def simulate(n, seed, shift=False):
    """Generate observational-looking synthetic firms and stochastic readiness outcomes.

    No real businesses, respondents, surveys or observations are used. The latent
    generator is not accessible to fitted models; the target is sampled stochastically.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    tech = rng.normal(size=n)
    innov = rng.normal(size=n)
    management = rng.normal(size=n)
    noise = rng.normal(size=(n, len(FEATURES)))
    core = np.column_stack([
        .62 * latent + .48 * tech, .68 * tech + .34 * latent,
        .53 * tech + .47 * management, .65 * management + .18 * latent,
        .68 * innov + .28 * management, .64 * innov + .27 * management,
        .63 * tech + .35 * management, .63 * latent + .30 * management,
    ])
    vals = expit(.70 * core + .67 * noise)
    if shift:
        vals[:, 0] = np.clip(vals[:, 0] - .13, .01, .99)
        vals[:, 6] = np.clip(vals[:, 6] - .17, .01, .99)
        vals[:, 7] = np.clip(vals[:, 7] - .12, .01, .99)
    sector = rng.choice(['Manufacturing', 'Services', 'Retail', 'Technology'],
                        n, p=[.33, .31, .22, .14])
    size = rng.choice(['SME', 'Large'], n, p=[.74, .26])
    df = pd.DataFrame(vals, columns=FEATURES)
    df['sector'], df['firm_size'] = sector, size
    z = (-3.0 + .82 * df['data_maturity'] + .90 * df['integration']
         + 1.20 * df['digital_skills'] + 1.00 * df['leadership']
         + .83 * df['innovation'] + .72 * df['entrepreneurial_agility']
         + 1.17 * df['cyber_governance'] + .80 * df['investment_capacity']
         + 1.80 * (df['data_maturity'] - .5) * (df['digital_skills'] - .5)
         + 1.0 * (df['innovation'] - .5) * (df['leadership'] - .5)
         - .34 * (sector == 'Retail') + .25 * (sector == 'Technology')
         - .24 * (size == 'SME'))
    if shift:
        z -= .25  # unexplained regime shift; not visible as a predictor
    p = expit(z)
    df['readiness_outcome'] = rng.binomial(1, p)
    df.insert(0, 'simulated_firm_id', [f'SIM-{seed}-{i:05d}' for i in range(n)])
    return df


def build_models():
    pre = ColumnTransformer([
        ('num', StandardScaler(), FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ])
    return {
        'Logistic regression': Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1500, random_state=SEED))]),
        'Random forest': Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=180, min_samples_leaf=8,
                                                                               random_state=SEED, n_jobs=-1))]),
        'Gradient boosting': Pipeline([('pre', pre), ('clf', HistGradientBoostingClassifier(max_iter=130,
                                                                                            max_leaf_nodes=15,
                                                                                            l2_regularization=4.0,
                                                                                            learning_rate=.065,
                                                                                            random_state=SEED))]),
    }


def metrics(y, p):
    pred = (p >= .5).astype(int)
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)))


def actions_for_policy(df, scores, policy):
    n = len(df)
    safe = ((df['data_maturity'].to_numpy() >= .42) &
            (df['cyber_governance'].to_numpy() >= .45) &
            (df['investment_capacity'].to_numpy() >= .38))
    pilot_allowed = df['investment_capacity'].to_numpy() >= .28
    if policy == 'Score threshold':
        action = np.where(scores >= .5, 'launch', 'defer')
    elif policy == 'Knowledge rules only':
        action = np.where(safe, 'launch', np.where(pilot_allowed, 'pilot', 'defer'))
    else:
        expected = np.column_stack([22 - 20 * scores, 9 - 3 * scores, 1 + 11 * scores])
        if policy == 'K-DRIVE':
            expected[~safe, 0] = np.inf
            expected[~pilot_allowed, 1] = np.inf
        action = np.array(['launch', 'pilot', 'defer'])[np.argmin(expected, axis=1)]
    return action


def evaluate_policy(df, scores, policy):
    action = actions_for_policy(df, scores, policy)
    ready = df['readiness_outcome'].to_numpy().astype(bool)
    gated = ((df['data_maturity'].to_numpy() < .42) |
             (df['cyber_governance'].to_numpy() < .45) |
             (df['investment_capacity'].to_numpy() < .38))
    # Predeclared, illustrative scenario loss units, not enterprise cash flows.
    realized = np.where(action == 'launch', np.where(ready, 2., 22.),
                        np.where(action == 'pilot', np.where(ready, 6., 9.),
                                 np.where(ready, 12., 1.)))
    realized = realized + 8. * ((action == 'launch') & gated)
    unsafe = (action == 'launch') & gated
    return action, realized, unsafe


def policy_summary(df, scores):
    result = {}
    for name in POLICIES:
        a, loss, unsafe = evaluate_policy(df, scores, name)
        result[name] = dict(mean_loss=float(loss.mean()), unsafe_launch_rate=float(unsafe.mean()),
                            launch_rate=float((a == 'launch').mean()),
                            pilot_rate=float((a == 'pilot').mean()),
                            defer_rate=float((a == 'defer').mean()),
                            total_unsafe_launches=int(unsafe.sum()))
    return result


def bootstrap_diff(x, y, iters=2000, seed=54321):
    """Paired percentile bootstrap of mean(x-y); relative to second comparator."""
    rng = np.random.default_rng(seed)
    n = len(x)
    contrasts = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        contrasts[i] = (x[idx] - y[idx]).mean()
    return [float(np.quantile(contrasts, .025)), float(np.quantile(contrasts, .975))]


def save_figures(y, all_scores, scores, df_test, policy_results, chosen):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 320})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for name, p in all_scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC {roc_auc_score(y,p):.3f})')
        ax[1].plot(recall, precision, lw=2, label=f'{name} (AP {average_precision_score(y,p):.3f})')
    ax[0].plot([0,1],[0,1],ls=':',color='gray',lw=1)
    ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='(a) Held-out discrimination')
    ax[1].axhline(np.mean(y), ls=':',color='gray',lw=1,label='Outcome prevalence')
    ax[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision–recall trade-off')
    for a in ax: a.set_xlim(0,1); a.set_ylim(0,1.02); a.grid(alpha=.14); a.legend(fontsize=7,loc='lower right')
    fig.savefig(OUT/'figure_1_predictive_validation.png',bbox_inches='tight')
    fig.savefig(OUT/'figure_1_predictive_validation.pdf',bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    labels = ['Threshold','Cost only','Rules only','K-DRIVE']
    vals = [policy_results[n]['mean_loss'] for n in POLICIES]
    unc = []
    rng = np.random.default_rng(914)
    for name in POLICIES:
        _, l, _ = evaluate_policy(df_test, scores, name)
        boots = np.array([l[rng.integers(0,len(l),len(l))].mean() for _ in range(1000)])
        unc.append([np.mean(l)-np.quantile(boots,.025), np.quantile(boots,.975)-np.mean(l)])
    yerr = np.array(unc).T
    ax[0].bar(labels, vals, yerr=yerr, capsize=4, alpha=.90)
    ax[0].set(ylabel='Mean simulated decision loss (units / firm)', title='(a) Decision quality (95% bootstrap CI)')
    ax[0].tick_params(axis='x',rotation=15)
    shares = np.array([[policy_results[n][f'{x}_rate'] for x in ['launch','pilot','defer']] for n in POLICIES])
    bottom=np.zeros(4)
    for i,(act, col) in enumerate([('Launch','#537caa'),('Pilot','#ddb469'),('Defer','#8aadb0')]):
        ax[1].bar(labels,shares[:,i]*100,bottom=bottom*100,label=act,color=col)
        bottom+=shares[:,i]
    ax[1].set(ylabel='Share of simulated firms (%)',title='(b) Recommended intervention mix',ylim=(0,100))
    ax[1].tick_params(axis='x',rotation=15)
    ax[1].legend(frameon=False,ncol=3,loc='upper center',fontsize=8)
    for a in ax: a.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'figure_2_policy_validation.png',bbox_inches='tight')
    fig.savefig(OUT/'figure_2_policy_validation.pdf',bbox_inches='tight')
    plt.close(fig)

    # Standalone programmatic graphical abstract; distinct from the paper's two numbered figures.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax=plt.subplots(figsize=(14,4),dpi=180)
    ax.set_xlim(0,14);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(.3,'SIMULATED\nENTERPRISE DATA\n8 capabilities + sector'),
           (3.85,'PREDICTIVE AI\nHeld-out model selection\nProbability of readiness'),
           (7.40,'KNOWLEDGE RULES\nData, cyber, budget\nFeasible-action gate'),
           (10.95,'DECISION\nLaunch / pilot / defer\nMeasured scenario loss')]
    cols=['#e8eef6','#e9f1e7','#fff0d8','#e5edf0']
    for (x,st),c in zip(boxes,cols):
        rect=FancyBboxPatch((x,.85),2.73,2.2,boxstyle='round,pad=0.20',edgecolor='#344258',facecolor=c,lw=1.7)
        ax.add_patch(rect); ax.text(x+1.365,1.96,st,ha='center',va='center',fontsize=9.4,fontweight='bold',linespacing=1.55)
    for x in [3.3,6.85,10.4]:
        ax.add_patch(FancyArrowPatch((x,1.95),(x+.52,1.95),arrowstyle='-|>',mutation_scale=18,lw=2,color='#344258'))
    ax.text(7,.26,'K-DRIVE | Controlled simulation; not field evidence',ha='center',fontsize=10,color='#4a5666')
    fig.savefig(OUT/'graphical_abstract.png',bbox_inches='tight',facecolor='white',dpi=300)
    fig.savefig(OUT/'graphical_abstract.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    data = simulate(6000, SEED)
    # Stratified, mutually disjoint partitions (train 60%, validation 20%, test 20%).
    train_val, test = train_test_split(data,test_size=.2,stratify=data['readiness_outcome'],random_state=SEED)
    train,val = train_test_split(train_val,test_size=.25,stratify=train_val['readiness_outcome'],random_state=SEED+1)
    assert len(train)==3600 and len(val)==1200 and len(test)==1200
    assert not (set(train.simulated_firm_id)&set(test.simulated_firm_id))
    models=build_models()
    val_brier={}
    prediction_rows=[]
    trained={}
    for name, model in models.items():
        model.fit(train[PREDICTORS],train['readiness_outcome'])
        val_brier[name] = brier_score_loss(val['readiness_outcome'],model.predict_proba(val[PREDICTORS])[:,1])
        trained[name]=model
    chosen = min(val_brier,key=val_brier.get)
    test_scores={name:m.predict_proba(test[PREDICTORS])[:,1] for name,m in trained.items()}
    y=test['readiness_outcome'].to_numpy()
    predictive={name:metrics(y,p) for name,p in test_scores.items()}
    probs=test_scores[chosen]
    policy=policy_summary(test,probs)
    losses={name:evaluate_policy(test,probs,name)[1] for name in POLICIES}
    ci_rule=bootstrap_diff(losses['K-DRIVE'],losses['Cost-aware (no rules)'])
    ci_score=bootstrap_diff(losses['K-DRIVE'],losses['Score threshold'])
    # Prospective sensitivity analysis to the assumed noncompliant-launch penalty.
    # This is descriptive policy stress-testing under synthetic rewards, not causal field data.
    safety_penalty_sensitivity = []
    for penalty in [0, 1, 2, 4, 8, 12, 20]:
        means = {}
        for name in POLICIES:
            a, v, unsafe = evaluate_policy(test, probs, name)
            baseline_losses = v - 8.0*unsafe.astype(float)
            means[name] = float((baseline_losses + penalty*unsafe.astype(float)).mean())
        safety_penalty_sensitivity.append({
            'penalty': penalty, 'policy_mean_loss': means,
            'kdrive_minus_no_rules': means['K-DRIVE'] - means['Cost-aware (no rules)']})

    # Feasibility robustness: policy threshold choices are illustrative, NOT estimated from data.
    threshold_sensitivity=[]
    true_ready = test['readiness_outcome'].to_numpy().astype(bool)
    reference_unsafe = ((test['data_maturity'].to_numpy()<.42)|
                        (test['cyber_governance'].to_numpy()<.45)|
                        (test['investment_capacity'].to_numpy()<.38))
    c = np.column_stack([22-20*probs,9-3*probs,1+11*probs])
    for delta in [-.05, 0.0, .05]:
        safe = ((test['data_maturity'].to_numpy()>=.42+delta)&
                (test['cyber_governance'].to_numpy()>=.45+delta)&
                (test['investment_capacity'].to_numpy()>=.38+delta))
        eligible_pilot = test['investment_capacity'].to_numpy()>=.28
        q=c.copy();q[~safe,0]=np.inf;q[~eligible_pilot,1]=np.inf
        chosen_a=np.array(['launch','pilot','defer'])[np.argmin(q,axis=1)]
        b=np.where(chosen_a=='launch',np.where(true_ready,2.,22.),
                   np.where(chosen_a=='pilot',np.where(true_ready,6.,9.),
                            np.where(true_ready,12.,1.)))
        violations=(chosen_a=='launch') & reference_unsafe
        threshold_sensitivity.append({
            'common_threshold_offset':delta,'mean_loss_at_reference_costs':float((b+8*violations).mean()),
            'reference_unsafe_launches':int(violations.sum()),
            'launch_share':float((chosen_a=='launch').mean())})

    # Shift data from an independent generator with worse data/cyber/budget and a latent base-rate change.
    stress=simulate(1500,SEED+2000,shift=True)
    p_stress=trained[chosen].predict_proba(stress[PREDICTORS])[:,1]
    stress_policy=policy_summary(stress,p_stress)
    stress_predictions=metrics(stress.readiness_outcome.to_numpy(),p_stress)
    result={
        'experiment':'synthetic controlled proof-of-concept, not field evidence',
        'seed':SEED,'train_n':len(train),'val_n':len(val),'test_n':len(test),'stress_n':len(stress),
        'readiness_prevalence_test':float(y.mean()),'selected_model':chosen,
        'validation_brier':{k:float(v) for k,v in val_brier.items()},
        'predictive_test':predictive,'policy_test':policy,
        'paired_95pct_bootstrap_ci_loss_diff_kdrive_minus_no_rules':ci_rule,
        'paired_95pct_bootstrap_ci_loss_diff_kdrive_minus_threshold':ci_score,
        'stress_selected_model':stress_predictions,'stress_policy':stress_policy,
        'penalty_sensitivity':safety_penalty_sensitivity,
        'threshold_sensitivity':threshold_sensitivity,
        'cost_parameters':{'launch_ready':2,'launch_not_ready':22,'pilot_ready':6,'pilot_not_ready':9,
                           'defer_ready':12,'defer_not_ready':1,'noncompliant_launch_penalty':8},
        'rules':{'launch_min_data':.42,'launch_min_cyber':.45,
                 'launch_min_investment':.38,'pilot_min_investment':.28},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    }
    data['set'] = 'training'
    data.loc[data.simulated_firm_id.isin(val.simulated_firm_id),'set']='validation'
    data.loc[data.simulated_firm_id.isin(test.simulated_firm_id),'set']='test'
    data.to_csv(OUT/'synthetic_enterprise_records.csv',index=False,float_format='%.8f')
    prediction_df=test[['simulated_firm_id','readiness_outcome']].copy()
    for name,p in test_scores.items(): prediction_df[name.lower().replace(' ','_')+'_probability']=p
    for name in POLICIES:
        act, loss, unsafe=evaluate_policy(test,probs,name)
        key=name.lower().replace(' ','_').replace('(','').replace(')','').replace('-','_')
        prediction_df[key+'_action']=act
        prediction_df[key+'_loss']=loss
        prediction_df[key+'_unsafe']=unsafe
    prediction_df.to_csv(OUT/'test_predictions_and_decisions.csv',index=False)
    stress.to_csv(OUT/'stress_synthetic_records.csv',index=False,float_format='%.8f')
    with open(OUT/'results.json','w') as f:json.dump(result,f,indent=2)
    pd.DataFrame([{'penalty':a['penalty'], 'kdrive_minus_no_rules':a['kdrive_minus_no_rules'], **a['policy_mean_loss']} for a in safety_penalty_sensitivity]).to_csv(OUT/'supp_penalty_sensitivity.csv',index=False)
    pd.DataFrame(threshold_sensitivity).to_csv(OUT/'supp_threshold_sensitivity.csv',index=False)
    save_figures(y,test_scores,probs,test,policy,chosen)
    print(json.dumps({k:result[k] for k in ['selected_model','readiness_prevalence_test',
                                            'predictive_test','policy_test',
                                            'paired_95pct_bootstrap_ci_loss_diff_kdrive_minus_no_rules',
                                            'stress_policy']},indent=2))

if __name__=='__main__':main()