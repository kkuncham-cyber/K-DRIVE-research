"""Independent observational survey illustration of K-DRIVE's prediction stage.

Source: Clim, A. (2026), DOI: 10.5281/zenodo.21235494,
GitHub release public_anonymised_dataset.csv, CC BY 4.0.

Caution: the 306 response records are individual survey respondents, many students;
we predict stated investment intention, NOT company readiness or investment success.
CSV is a documented projection of the upstream public data with no identifiers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, roc_curve, precision_recall_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
O = HERE/'outputs'; O.mkdir(exist_ok=True)
NAMES = ['ai_familiarity','ai_improves_operations','ai_data_handling_confidence','awareness_eu_ai_regulations', 'ai_integration_cost_perception','ai_process_transparency','ai_navigate_uncertainties']
CATS = ['data_access_challenges_code','workforce_preparation_code','sector_technology_code','student_code']
SEED = 62026

def setup():
    df=pd.read_csv(HERE/'data'/'survey_public_projection.csv')
    assert len(df)==306 and df['response_id'].is_unique
    assert df.investment_intention_code.value_counts().sort_index().to_dict()=={0:68,1:23,2:67,3:105,4:43}
    assert df.student_code.value_counts().sort_index().to_dict()=={0:132,1:174}
    for c in NAMES:
        df.loc[df[c]==15,c]=np.nan
    df['high_investment_intention']=(df.investment_intention_code>=3).astype(int)
    return df

def models():
    pre=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('standardize', StandardScaler())]),NAMES),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),CATS),
    ])
    return {
      'Prevalence-only':DummyClassifier(strategy='prior'),
      'Logistic regression':Pipeline([('pre',pre),('clf',LogisticRegression(C=.5,max_iter=2500,random_state=SEED))]),
      'Random forest':Pipeline([('pre',pre),('clf',RandomForestClassifier(n_estimators=240,min_samples_leaf=8,random_state=SEED,n_jobs=1))]),
      'Gradient boosting':Pipeline([('pre',pre),('clf',HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=8,l2_regularization=6.0,learning_rate=.05,random_state=SEED))]),
    }

def bootstrap_auc(y,p,n=2000):
    rng=np.random.default_rng(SEED+1); auc=[]; ap=[]
    for _ in range(n):
        ids=rng.integers(0,len(y),len(y)); yt=y[ids];pp=p[ids]
        if len(np.unique(yt))!=2:continue
        auc.append(roc_auc_score(yt,pp));ap.append(average_precision_score(yt,pp))
    return list(np.quantile(auc,[.025,.975])),list(np.quantile(ap,[.025,.975]))

def evaluate():
    df=setup(); predictors=NAMES+CATS
    train,test=train_test_split(df,test_size=.25,stratify=df.high_investment_intention,random_state=SEED)
    assert len(train)==229 and len(test)==77
    assert not set(train.response_id)&set(test.response_id)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+1)
    selection={}
    for name,model in models().items():
        p=cross_val_predict(model,train[predictors],train.high_investment_intention,cv=cv,method='predict_proba')[:,1]
        selection[name]=float(brier_score_loss(train.high_investment_intention,p))
    selected=min(selection,key=selection.get)
    result={}
    pred=pd.DataFrame({'response_id':test.response_id.values,'reported_investment_intention':test.high_investment_intention.values,'is_student':test.student_code.values})
    probs={}
    for name,model in models().items():
        model.fit(train[predictors],train.high_investment_intention)
        p=model.predict_proba(test[predictors])[:,1]
        probs[name]=p
        pred[name.lower().replace(' ','_')+'_score']=p
        y=test.high_investment_intention.to_numpy(); scores=(p>=.5).astype(int)
        result[name]={
            'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'f1':float(f1_score(y,scores,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,scores))
        }
    p=probs[selected]; y=test.high_investment_intention.to_numpy()
    auc_ci,ap_ci=bootstrap_auc(y,p)
    # Descriptive *decision flags* only; no observed payoffs, not an economic loss estimate.
    # Feasibility screens defined for demonstration; no claim of expert-validated thresholds.
    low_governance=test.ai_data_handling_confidence.to_numpy()<5
    data_problem=test.data_access_challenges_code.to_numpy()==3
    no_preparation=test.workforce_preparation_code.to_numpy()==0
    flagged=low_governance|data_problem|no_preparation
    decide=(p>=.5).astype(bool)
    action=np.where(decide & (~flagged),'recommend_invest',np.where(decide & flagged,'recommend_pilot','monitor'))
    pred['illustrative_policy']=action
    pred['flagged_missing_capability']=flagged
    details={
      'data_origin':'Clim 2026 open anonymized respondent survey (not enterprise outcome records)',
      'target':'investment intention considering_it or yes_definitely (NOT realized investment)',
      'n':len(df),'n_train':len(train),'n_test':len(test),
      'n_student':int(df.student_code.sum()),'positive_n':int(df.high_investment_intention.sum()),
      'selected_by_5fold_train_brier':selected,'cv_brier':selection,
      'heldout':result,'selected_auc_bootstrap_95ci':auc_ci,'selected_ap_bootstrap_95ci':ap_ci,
      'test_positive_n':int(y.sum()),'selected_recommend_invest':int(np.sum(action=='recommend_invest')),
      'selected_recommend_pilot':int(np.sum(action=='recommend_pilot')),
      'selected_monitor':int(np.sum(action=='monitor')),
      'test_nonstudent_n':int((test.student_code==0).sum()),
      'conditional_rule_override_n':int(np.sum(decide&flagged)),
    }
    pred.to_csv(O/'survey_holdout_predictions.csv',index=False)
    pd.DataFrame([{'model':k,**v} for k,v in result.items()]).to_csv(O/'survey_table_predictive.csv',index=False)
    with (O/'survey_results.json').open('w') as f: json.dump(details,f,indent=2)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.2),layout='constrained')
    for name,pp in probs.items():
        if name=='Prevalence-only':continue
        fpr,tpr,_=roc_curve(y,pp); recpr, rec,_=precision_recall_curve(y,pp)
        axs[0].plot(fpr,tpr,lw=2,label=f'{name} (AUC {roc_auc_score(y,pp):.2f})')
        axs[1].plot(rec,recpr,lw=2,label=f'{name} (AP {average_precision_score(y,pp):.2f})')
    axs[0].plot([0,1],[0,1],color='gray',ls=':',lw=1,label='Chance')
    axs[1].axhline(y.mean(),color='gray',ls=':',lw=1,label='Test prevalence')
    axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='(a) ROC: held-out intentions',xlim=(0,1),ylim=(0,1.02))
    axs[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision–recall',xlim=(0,1),ylim=(0,1.02))
    for ax in axs: ax.grid(alpha=.15);ax.legend(fontsize=7,loc='lower right')
    fig.savefig(O/'figure_1_real_survey.png',bbox_inches='tight');fig.savefig(O/'figure_1_real_survey.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(details,indent=2))
    return details

if __name__=='__main__':evaluate()