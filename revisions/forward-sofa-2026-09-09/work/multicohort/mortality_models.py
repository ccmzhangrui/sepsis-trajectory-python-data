"""Secondary hospital-mortality models. Never a deterioration-time surrogate."""
from pathlib import Path
import json,os
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve,confusion_matrix
from scipy.optimize import minimize
from scipy.special import expit
W=Path('work/multicohort');P=W/'private';R=W/'results'
SEED=20260908

def pipeline():
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=SEED))
def threshold(y,p):
 fpr,tpr,t=roc_curve(y,p);m=np.isfinite(t);return float(t[m][np.argmax((tpr-fpr)[m])])
def metrics(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])
def calibration(y,p):
 z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
 def loss(b):
  eta=b[0]+b[1]*z;return np.sum(np.logaddexp(0,eta)-y*eta)
 fit=minimize(loss,[0,1],method='BFGS');return {'calibration_intercept':float(fit.x[0]),'calibration_slope':float(fit.x[1])}
def run(cohort_name,features,cohort,site_folds=False):
 assert features.index.equals(cohort.index) and cohort.subject_key.is_unique
 y=cohort.death.to_numpy(dtype=int);groups=cohort.hospitalid.to_numpy() if site_folds else None
 names={'demographics':['age_topcoded90','female'], 'latest':['age_topcoded90','female']+[v+'_last' for v in ['hr','rr','spo2','temp']],
 'history':['age_topcoded90','female']+[v+'_'+s for v in ['hr','rr','spo2','temp'] for s in ['last','mean','sd','slope','n_hours']]}
 # SOFA comparator only where original SOFA was actually reconstructed.
 if 'sofa_last' in features:names['sofa_history']=names['history']+[k for k in features if k.startswith('sofa_')]
 splits=list(GroupKFold(5,shuffle=True,random_state=SEED).split(features,y,groups) if site_folds else StratifiedKFold(5,shuffle=True,random_state=SEED).split(features,y))
 preds={name:np.zeros(len(y)) for name in names};alarms={name:np.zeros(len(y),dtype=bool) for name in names};folds=np.zeros(len(y),dtype=int);ths=[]
 for fold,(tr,te) in enumerate(splits,1):
  assert set(cohort.subject_key.iloc[tr]).isdisjoint(cohort.subject_key.iloc[te])
  if site_folds:assert set(groups[tr]).isdisjoint(groups[te])
  inner=list(GroupKFold(3,shuffle=True,random_state=SEED+fold).split(features.iloc[tr],y[tr],groups[tr]) if site_folds else StratifiedKFold(3,shuffle=True,random_state=SEED+fold).split(features.iloc[tr],y[tr]))
  for name,cols in names.items():
   X=features[cols];ip=np.zeros(len(tr))
   for it,iv in inner:
    m=pipeline();m.fit(X.iloc[tr[it]],y[tr[it]]);ip[iv]=m.predict_proba(X.iloc[tr[iv]])[:,1]
   th=threshold(y[tr],ip);m=pipeline();m.fit(X.iloc[tr],y[tr]);preds[name][te]=m.predict_proba(X.iloc[te])[:,1];alarms[name][te]=preds[name][te]>=th
   ths.append({'cohort':cohort_name,'model':name,'fold':fold,'threshold':th,'n_train':len(tr),'n_test':len(te),'deaths_test':int(y[te].sum())})
  folds[te]=fold;print(cohort_name,'fold',fold,'complete',flush=True)
 gen=np.random.default_rng(SEED+100);boot={name:[] for name in names};pairs=[]
 if site_folds:
  units=np.unique(groups);indices={g:np.flatnonzero(groups==g) for g in units}
 else:neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1)
 for _ in range(1000):
  ix=np.concatenate([indices[g] for g in gen.choice(units,len(units),replace=True)]) if site_folds else np.r_[gen.choice(neg,len(neg),True),gen.choice(pos,len(pos),True)]
  for name in names:boot[name].append(metrics(y[ix],preds[name][ix]))
  pairs.append(boot['history'][-1]-boot['latest'][-1])
 rows=[]
 for name in names:
  est=metrics(y,preds[name]);ci=np.quantile(boot[name],[.025,.975],axis=0);tn,fp,fn,tp=confusion_matrix(y,alarms[name]).ravel()
  row={'cohort':cohort_name,'model':name,'n':len(y),'deaths':int(y.sum()),'features':len(names[name]),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'npv':tn/(tn+fn),**calibration(y,preds[name])}
  for j,key in enumerate(['auroc','auprc','brier']):row.update({key:est[j],key+'_lo':ci[0,j],key+'_hi':ci[1,j]})
  rows.append(row)
 pd.DataFrame(rows).to_csv(R/f'{cohort_name}_mortality_metrics.csv',index=False);pd.DataFrame(ths).to_csv(R/f'{cohort_name}_mortality_thresholds.csv',index=False)
 delta=metrics(y,preds['history'])-metrics(y,preds['latest']);ci=np.quantile(pairs,[.025,.975],axis=0)
 pd.DataFrame([{'metric':k,'difference':delta[j],'lo':ci[0,j],'hi':ci[1,j]} for j,k in enumerate(['auroc','auprc','brier'])]).to_csv(R/f'{cohort_name}_mortality_paired.csv',index=False)
 out=pd.DataFrame({'death':y,'fold':folds,**preds},index=features.index);out.to_pickle(P/f'{cohort_name}_mortality_oof.pkl')
 audit={'cohort':cohort_name,'n':len(y),'deaths':int(y.sum()),'predictor_registry':names,'seed':SEED,'outer_folds':5,'inner_folds':3,'fit_count':len(names)*20,'split_unit':'hospital' if site_folds else 'patient','bootstrap_unit':'hospital' if site_folds else 'stratified patient','bootstrap_repeats':1000,'CI_scope':'conditional on frozen out-of-fold predictions; no full model refit','endpoint':'recorded hospital mortality after24h among patients still at risk','all_patients_have_one_oof_prediction':True}
 (R/f'{cohort_name}_mortality_audit.json').write_text(json.dumps(audit,indent=2));print(pd.DataFrame(rows).to_json(orient='records'),flush=True)
 return names
if __name__=='__main__':
 import sys
 name=sys.argv[1]
 if name=='mimiciii':
  F=pd.read_pickle(W/'mimiciii/private/features_24h.pkl');C=pd.read_pickle(W/'mimiciii/private/cohort.pkl').set_index('icustay_id').loc[F.index]
  C=C.assign(subject_key=C.subject_id,death=C.hospital_expire_flag.astype(int));assert ((C.outtime-C.intime).dt.total_seconds()>86400).all();run(name,F,C)
 elif name=='eicu':
  F=pd.read_pickle(P/'eicu_features.pkl');C=pd.read_pickle(P/'eicu_cohort.pkl').set_index('patientunitstayid').loc[F.index]
  C=C.assign(subject_key=C.uniquepid,death=C.hospitaldischargestatus.eq('Expired').astype(int));run(name,F,C,site_folds=True)
