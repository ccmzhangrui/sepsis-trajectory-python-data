"""Real-data retrospective mortality sensitivity analysis; no synthetic fallback.

Day labels are nominal extraction slots, NOT verified ICU landmarks.
Default input is the user-supplied workbook. Cache is an optional local speed-up.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse, json, hashlib, sys, platform, warnings
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix

SEED=20240908
ap=argparse.ArgumentParser()
ap.add_argument('--input',required=True)
ap.add_argument('--cache')
ap.add_argument('--output',default='work/reanalysis/results')
args=ap.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
source=Path(args.input); sha=hashlib.sha256(source.read_bytes()).hexdigest()
d=pd.read_pickle(args.cache) if args.cache else pd.read_excel(source,na_values=['NULL'])
assert d.shape==(51992,215), 'Unexpected workbook; review schema before proceeding'
d=d.replace('NULL',np.nan)
d=d.loc[pd.to_numeric(d.sepsis3,errors='coerce').eq(1)].copy()
assert len(d)==25862 and d.subject_id.is_unique and d.stay_id.is_unique
d=d.sort_values('subject_id').reset_index(drop=True)
y=pd.to_numeric(d.is_hosp_dead,errors='raise').to_numpy(dtype=int)
assert set(y)=={0,1} and y.sum()==4416
base=pd.DataFrame({'age':pd.to_numeric(d.age,errors='coerce'),
 'female':d.gender.map({'F':1.,'M':0.})})
assert base.notna().all().all()
# Fixed broad plausibility limits selected before model fitting; no outcome-dependent cleaning.
variables={'temperature':('体温(℃)',25,45),'heart_rate':('心率(次/min)',20,300),
 'respiratory_rate':('呼吸频率(次/min)',1,80),'spo2':('脉搏氧饱和度(%)',40,100)}
quality=[]; registry=[]; matrices={'demographic':base.copy()}
for bins in (1,2,4):
 last=base.copy();hist=base.copy()
 for key,(prefix,lo,hi) in variables.items():
  cols=[f'{prefix}_Day1_{h:02d}:00' for h in range(0,bins*6,6)]
  a=d[cols].apply(pd.to_numeric,errors='coerce');bad=(a<lo)|(a>hi)
  for col in cols:
   if bins==4:quality.append({'column':col,'nonmissing_before':int(a[col].notna().sum()),'outside_range':int(bad[col].sum()),'lower':lo,'upper':hi})
  a=a.mask(bad);valid=a.notna().to_numpy();v=a.to_numpy();n=valid.sum(axis=1)
  latest=a.ffill(axis=1).iloc[:,-1].to_numpy()
  last[key+'_last']=latest;hist[key+'_last']=latest
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   mean=np.nanmean(v,axis=1);std=np.nanstd(v,axis=1,ddof=1)
  times=np.arange(bins)*6.;t=np.where(valid,times,np.nan)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   tc=t-np.nanmean(t,axis=1)[:,None]
  denom=np.nansum(tc*tc,axis=1)
  slope=np.divide(np.nansum(tc*(v-mean[:,None]),axis=1),denom,out=np.full(len(d),np.nan),where=denom>0)
  hist[key+'_mean']=mean;hist[key+'_sd']=std;hist[key+'_slope']=slope;hist[key+'_count']=n
  registry.append({'nominal_bins':bins,'variable':key,'source_columns':cols,'latest_rule':'last nonmissing among included columns only','time_interpretation':'unverified nominal extraction slots; no ICU landmark claim'})
 matrices[f'last_{bins}bin']=last
 if bins>1:matrices[f'history_{bins}bin']=hist

def model():
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
  HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=SEED))
outer=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(base,y))
fold=np.full(len(y),-1);pred={};decision={};thresholds=[]
for name,X in matrices.items():
 p=np.full(len(y),np.nan);cl=np.full(len(y),-1)
 for k,(tr,te) in enumerate(outer):
  assert set(d.subject_id.iloc[tr]).isdisjoint(set(d.subject_id.iloc[te]))
  ip=np.full(len(tr),np.nan)
  for itr,ite in StratifiedKFold(3,shuffle=True,random_state=SEED+k+1).split(X.iloc[tr],y[tr]):
   m=model();m.fit(X.iloc[tr[itr]],y[tr[itr]]);ip[ite]=m.predict_proba(X.iloc[tr[ite]])[:,1]
  fpr,tpr,ts=roc_curve(y[tr],ip);valid=np.isfinite(ts);ix=np.flatnonzero(valid)[np.argmax((tpr-fpr)[valid])];threshold=float(ts[ix])
  m=model();m.fit(X.iloc[tr],y[tr]);p[te]=m.predict_proba(X.iloc[te])[:,1];cl[te]=(p[te]>=threshold).astype(int);fold[te]=k
  thresholds.append({'model':name,'fold':k,'threshold':threshold,'training_n':len(tr),'heldout_n':len(te),'heldout_events':int(y[te].sum())})
  print(f'{name} fold {k+1}/5 complete',flush=True)
 assert np.isfinite(p).all() and (cl>=0).all()
 pred[name]=p;decision[name]=cl
 np.savez_compressed(out/'local_predictions.npz',y=y,fold=fold,**pred)

def metrics(yy,pp):return {'auroc':roc_auc_score(yy,pp),'auprc':average_precision_score(yy,pp),'brier':brier_score_loss(yy,pp)}
def calibration(yy,pp):
 z=np.log(np.clip(pp,1e-8,1-1e-8)/np.clip(1-pp,1e-8,1))
 def fun(b):
  eta=b[0]+b[1]*z;return np.mean(np.logaddexp(0,eta)-yy*eta)
 fit=minimize(fun,[0.,1.],method='BFGS',tol=1e-7)
 assert fit.success,fit.message
 offset=minimize(lambda b:np.mean(np.logaddexp(0,b[0]+z)-yy*(b[0]+z)),[0.],method='BFGS',tol=1e-7)
 assert offset.success,offset.message
 return {'calibration_intercept_joint':float(fit.x[0]),'calibration_slope':float(fit.x[1]),'calibration_in_large':float(offset.x[0])}
rows=[];boots={name:{k:[] for k in ('auroc','auprc','brier')} for name in pred}
rng=np.random.default_rng(SEED+100);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
for b in range(1000):
 ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
 for name,p in pred.items():
  for key,val in metrics(y[ix],p[ix]).items():boots[name][key].append(val)
 if (b+1)%100==0:print(f'bootstrap {b+1}/1000',flush=True)
for name,p in pred.items():
 row={'model':name,'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),'raw_feature_count':matrices[name].shape[1],**metrics(y,p),**calibration(y,p)}
 tn,fp,fn,tp=confusion_matrix(y,decision[name],labels=[0,1]).ravel()
 for key,val in {'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'npv':tn/(tn+fn)}.items():row[key]=float(val)
 for key in boots[name]:row[key+'_lo'],row[key+'_hi']=np.quantile(boots[name][key],[.025,.975]).tolist()
 rows.append(row)
pairs=[]
for bins in (2,4):
 for key in ('auroc','auprc','brier'):
  a=f'history_{bins}bin';b=f'last_{bins}bin';v=np.array(boots[a][key])-np.array(boots[b][key]);point=metrics(y,pred[a])[key]-metrics(y,pred[b])[key]
  pairs.append({'comparison':a+' minus '+b,'metric':key,'difference':point,'lo':float(np.quantile(v,.025)),'hi':float(np.quantile(v,.975))})
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
pd.DataFrame(pairs).to_csv(out/'paired_differences.csv',index=False)
pd.DataFrame(thresholds).to_csv(out/'thresholds.csv',index=False)
pd.DataFrame(quality).to_csv(out/'cleaning_audit.csv',index=False)
pd.DataFrame([{'model':name,'feature':col,'missing_n':int(X[col].isna().sum())} for name,X in matrices.items() for col in X]).to_csv(out/'feature_missingness.csv',index=False)
# Patient identifiers and predictions remain local and are excluded from submission archives.
pd.DataFrame({'subject_id':d.subject_id,'stay_id':d.stay_id,'fold':fold}).to_csv(out/'LOCAL_ONLY_fold_assignment.csv',index=False)
(out/'feature_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
summary={'source_sha256':sha,'source_rows':51992,'sepsis_records':len(y),'events':int(y.sum()),'endpoint':'recorded in-hospital mortality (is_hosp_dead)','seed':SEED,'bootstrap_replicates':1000,'ci_scope':'stratified patient resampling conditional on fitted out-of-fold predictions; not full model-refit uncertainty','time_scope':'retrospective nominal-column restriction, NOT verified ICU-landmark prediction','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'metrics':rows,'paired':pairs}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
print(json.dumps(summary,indent=2),flush=True)
