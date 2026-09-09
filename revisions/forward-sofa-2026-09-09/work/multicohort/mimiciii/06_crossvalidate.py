from pathlib import Path
import json,time,os
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve,confusion_matrix
from gbtm import CensoredGBTM

W=Path('work/multicohort/mimiciii');P=W/'private';R=W/'results';rng=np.random.default_rng(20260908)
Y=pd.read_pickle(P/'trajectory_matrix.pkl');F=pd.read_pickle(P/'features_24h.pkl');assert Y.index.equals(F.index)
times=np.asarray(Y.columns,dtype=float);yarray=Y.to_numpy();feature_audit=json.loads((W/'feature_audit.json').read_text())
no_sofa=feature_audit['no_sofa_predictors'];features=F.to_numpy(dtype=float,na_value=np.nan);independent=F[no_sofa].to_numpy(dtype=float,na_value=np.nan)
names=['full_window_posterior_diagnostic','truncated_posterior_12h','truncated_posterior_24h','sofa_without_posterior','no_sofa_inputs']
fit_records=[];fit_count=0
def make_blocks(train,test,seed):
    full=CensoredGBTM(k=3,starts=3,max_iter=150,seed=seed).fit(yarray[train],times)
    means=full.mean_curve(times);target=int(means[:,-1].argmax());assert means[target,-1]>means[target,0]
    ptrain=full.predict_proba(yarray[train],times);ptest=full.predict_proba(yarray[test],times)
    gt=(ptrain.argmax(axis=1)==target).astype(int);gv=(ptest.argmax(axis=1)==target).astype(int)
    btrain={'full_window_posterior_diagnostic':np.column_stack([features[train],ptrain]),'sofa_without_posterior':features[train],'no_sofa_inputs':independent[train]}
    btest={'full_window_posterior_diagnostic':np.column_stack([features[test],ptest]),'sofa_without_posterior':features[test],'no_sofa_inputs':independent[test]}
    record={'seed':seed,'n_train':len(train),'n_test':len(test),'target_group':target+1,'full_window_group_curves':means.tolist(),'full_fit_converged':full.best_start['em_converged'],'prefix':[]}
    for cutoff in [12,24]:
        mask=times<=cutoff;early=yarray[train][:,mask];ok=np.isfinite(early).sum(axis=1)>=2
        earlymodel=CensoredGBTM(k=3,degree=1 if cutoff==12 else 2,starts=3,max_iter=150,seed=seed+cutoff).fit(early[ok],times[mask])
        key='truncated_posterior_'+str(cutoff)+'h'
        btrain[key]=np.column_stack([features[train],earlymodel.predict_proba(early,times[mask])])
        btest[key]=np.column_stack([features[test],earlymodel.predict_proba(yarray[test][:,mask],times[mask])])
        record['prefix'].append({'cutoff':cutoff,'n_fit':int(ok.sum()),'converged':earlymodel.best_start['em_converged']})
    fit_records.append(record)
    return gt,gv,btrain,btest
def fit_predict(x,y,z,seed):
    global fit_count
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
      HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5,random_state=seed,early_stopping=False))
    model.fit(x,y);fit_count+=1;return model.predict_proba(z)[:,1]
def threshold(y,p):
    fpr,tpr,th=roc_curve(y,p);score=tpr-fpr;score[~np.isfinite(th)]=-np.inf
    return float(th[np.argmax(score)])
outer=KFold(n_splits=5,shuffle=True,random_state=20260908)
prob={name:np.full(len(Y),np.nan) for name in names};alarm={name:np.zeros(len(Y),dtype=bool) for name in names}
truth=np.full(len(Y),-1);fold_id=np.full(len(Y),-1);threshold_records=[]
for fold,(train,test) in enumerate(outer.split(Y),1):
    started=time.time();print(json.dumps({'start_outer_fold':fold}),flush=True)
    inner=KFold(n_splits=3,shuffle=True,random_state=20260908+fold)
    ivp={name:[] for name in names};ivy=[]
    for j,(it,iv) in enumerate(inner.split(train),1):
        a,b=train[it],train[iv];gt,gv,bt,bv=make_blocks(a,b,20260908+fold*100+j)
        ivy.append(gv)
        for name in names:ivp[name].append(fit_predict(bt[name],gt,bv[name],20260908+fold*100+j))
    th={name:threshold(np.concatenate(ivy),np.concatenate(ivp[name])) for name in names}
    gt,gv,bt,bv=make_blocks(train,test,20260908+fold*100)
    truth[test]=gv;fold_id[test]=fold
    for name in names:
        prob[name][test]=fit_predict(bt[name],gt,bv[name],20260908+fold*100)
        alarm[name][test]=prob[name][test]>=th[name]
        threshold_records.append({'fold':fold,'variant':name,'inner_youden_threshold':th[name]})
    print(json.dumps({'outer_fold':fold,'test_n':len(test),'test_phenotype_n':int(gv.sum()),'seconds':round(time.time()-started,1)}),flush=True)
assert (truth>=0).all() and all(np.isfinite(p).all() for p in prob.values())
indices=[np.flatnonzero(truth==value) for value in [0,1]]
bootstrap={name:[] for name in names};deltas=[]
for b in range(1000):
    idx=np.r_[rng.choice(indices[0],len(indices[0]),replace=True),rng.choice(indices[1],len(indices[1]),replace=True)]
    by={}
    for name in names:
        p=prob[name][idx];y=truth[idx]
        vals=[roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)]
        bootstrap[name].append(vals);by[name]=vals[0]
    deltas.append(by['truncated_posterior_24h']-by['no_sofa_inputs'])
rows=[]
for name in names:
    tn,fp,fn,tp=confusion_matrix(truth,alarm[name]).ravel();ci=np.quantile(bootstrap[name],[.025,.975],axis=0)
    rows.append({'variant':name,'n':len(Y),'high_increasing_phenotype_n':int(truth.sum()),'auroc':roc_auc_score(truth,prob[name]),'auroc_ci_low':ci[0,0],'auroc_ci_high':ci[1,0],
    'auprc':average_precision_score(truth,prob[name]),'auprc_ci_low':ci[0,1],'auprc_ci_high':ci[1,1],
    'brier':brier_score_loss(truth,prob[name]),'brier_ci_low':ci[0,2],'brier_ci_high':ci[1,2],
    'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'threshold_policy':'Youden from3innerfolds, selected before outertest',
    'interpretation':'Deliberately circular diagnostic, not prospective validation' if name=='full_window_posterior_diagnostic' else '24h prediction of fold-defined high-and-increasing SOFA phenotype; not independent clinical-event warning'})
pd.DataFrame(rows).to_csv(R/'trajectory_classification_metrics.csv',index=False)
pd.DataFrame(threshold_records).to_csv(R/'inner_thresholds.csv',index=False)
private=pd.DataFrame({'icustay_id':Y.index,'fold':fold_id,'target':truth,**prob})
private.to_pickle(P/'trajectory_oof_predictions.pkl');os.chmod(P/'trajectory_oof_predictions.pkl',0o600)
audit={'n':len(Y),'target_positive':int(truth.sum()),'classifier_fits':fit_count,'gbtm_fits':len(fit_records)*3,'bootstrap_repeats':1000,
 'target':'Highest fitted60h SOFA class of a training-fold3-class censored-normal GBTM; required fitted6-to60h increase in each fold.',
 'selected_three_class_scope':'Prespecified comparison with original3-class framing. Five classes had lower descriptive BIC.',
 'outer_folds':5,'inner_folds':3,'split_unit':'unique patient','fold_records':fit_records,
 'paired_delta_auroc_24h_prefix_vs_no_sofa':{'estimate':rows[2]['auroc']-rows[4]['auroc'],'ci':np.quantile(deltas,[.025,.975]).tolist()},
 'ci_scope':'Patient bootstrap of fixed pooled out-of-fold labels/predictions, conditional on fitted GBTMs and classifiers; no complete pipeline refit.',
 'warning_time':'No clinical-event warning time computed by this phenotype classifier.'}
(R/'crossvalidation_audit.json').write_text(json.dumps(audit,indent=2));print(pd.DataFrame(rows).to_json(orient='records'),flush=True)
