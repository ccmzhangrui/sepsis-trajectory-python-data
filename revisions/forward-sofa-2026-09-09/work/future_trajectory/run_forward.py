"""Forward-only SOFA trajectory labels and genuine 12/24-hour prediction origins."""
from pathlib import Path
import sys, json, hashlib, time, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, confusion_matrix, log_loss, accuracy_score
from scipy.stats import norm
sys.path.insert(0,str(Path('work/multicohort/mimiciii').resolve()))
from gbtm import CensoredGBTM
sys.path.insert(0,str(Path('work/multicohort').resolve()))
from time_risk_analysis import features
from figure_inference import WeightedMetrics,holm
W=Path('work/future_trajectory');P=W/'private';R=W/'results'
P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
SEED=20260919
NAMES=['history','latest_sofa','sofa_slope','prefix_added','prefix_only']
history=['age','female']+[v+'_'+a for v in ['hr','rr','temp','spo2'] for a in ['last','mean','sd','slope','n_hours']]

def main(t,mode):
 tag=f'{mode}_{t}h';start=time.time()
 c=pd.read_pickle('work/multicohort/mimiciii/private/cohort.pkl').set_index('icustay_id').sort_values('subject_id')
 c['exit_h']=(c.outtime-c.intime).dt.total_seconds()/3600;c['death_exit']=c.outtime.eq(c.deathtime)
 c['fold']=0
 for fold,(_,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(c),1):c.iloc[te,c.columns.get_loc('fold')]=fold
 assert len(c)==4856 and c.subject_id.is_unique
 s=pd.read_pickle('work/multicohort/mimiciii/private/sofa_hourly.pkl')
 s=s.loc[s.observed_components.ge(4)].copy()
 ymat=s.pivot(index='icustay_id',columns='hour',values='sofa_total').reindex(c.index)
 prefix=list(range(6,t+1,6));future=list(range(t+6 if mode=='main' else t+24,61,6))
 assert max(prefix)<min(future)
 if mode!='main':assert min(future)-24>=t
 ready=c.recognition_hour.le(t)&c.exit_h.gt(t);prefixok=ymat[prefix].notna().sum(axis=1).ge(2)
 risk=c.loc[ready&prefixok].copy();yp=ymat.loc[risk.index,prefix].to_numpy(float);yf=ymat.loc[risk.index,future].to_numpy(float)
 assess=np.isfinite(yf).sum(axis=1)>=3
 v=pd.read_pickle('work/multicohort/private/time_risk_vitals.pkl')
 f=features(t,c,v,s).loc[risk.index]
 f.to_pickle(P/(tag+'_features.pkl'))
 pd.DataFrame(yf,index=risk.index,columns=future).to_pickle(P/(tag+'_future_scores.pkl'))
 missing=risk.loc[~assess]
 flow=dict(origin_h=t,mode=mode,source_n=len(c),dead_by_origin=int((c.death_exit&c.exit_h.le(t)).sum()),other_exit_by_origin=int((~c.death_exit&c.exit_h.le(t)).sum()),present_not_recognized=int((~c.recognition_hour.le(t)&c.exit_h.gt(t)).sum()),recognized_present_n=int(ready.sum()),insufficient_prefix_n=int((ready&~prefixok).sum()),prefix_eligible_n=len(risk),future_evaluable_n=int(assess.sum()),future_unassessable_n=int((~assess).sum()),unassessable_death_by60=int((missing.death_exit&missing.exit_h.le(60)).sum()),unassessable_other_exit_by60=int((~missing.death_exit&missing.exit_h.le(60)).sum()),unassessable_recording_gap=int(missing.exit_h.gt(60).sum()),evaluable_death_by60=int((risk.death_exit&risk.exit_h.le(60)&assess).sum()),evaluable_other_exit_by60=int((~risk.death_exit&risk.exit_h.le(60)&assess).sum()),prefix_grid=prefix,future_grid=future)
 (R/(tag+'_flow.json')).write_text(json.dumps(flow,indent=2));print(tag,flow,flush=True)
 n=len(risk);truth=np.full(n,-1);risingtruth=np.full(n,-1)
 prob={k:np.full((n,3),np.nan) for k in NAMES};alarms={k:np.zeros((n,3),bool) for k in NAMES}
 riseprob={k:np.full(n,np.nan) for k in NAMES};risealarm={k:np.zeros(n,bool) for k in NAMES}
 audits=[];cuts=[];fits=0

 def one(tr,te,seed):
  nonlocal fits
  et=tr[assess[tr]]
  targetmodel=CensoredGBTM(k=3,degree=2,starts=3,max_iter=500,seed=seed).fit(yf[et],future)
  prefmodel=CensoredGBTM(k=3,degree=1 if t==12 else 2,starts=3,max_iter=500,seed=seed+37).fit(yp[tr],prefix)
  if not targetmodel.best_start['em_converged'] or not prefmodel.best_start['em_converged']:raise RuntimeError('GBTM failed convergence; retain fit audit and review')
  gt=targetmodel.predict_proba(yf[et],future).argmax(axis=1)
  gv=targetmodel.predict_proba(yf[te],future).argmax(axis=1);gv[~assess[te]]=-1
  assert set(gt)=={0,1,2}
  pp= prefmodel.predict_proba(yp[et],prefix);pv=prefmodel.predict_proba(yp[te],prefix)
  x=f.iloc[et];z=f.iloc[te]
  bx={'history':x[history].to_numpy(),'latest_sofa':x[history+['sofa_last']].to_numpy(),'sofa_slope':x[history+['sofa_last','sofa_slope']].to_numpy(),'prefix_added':np.c_[x[history+['sofa_last']],pp],'prefix_only':pp}
  bz={'history':z[history].to_numpy(),'latest_sofa':z[history+['sofa_last']].to_numpy(),'sofa_slope':z[history+['sofa_last','sofa_slope']].to_numpy(),'prefix_added':np.c_[z[history+['sofa_last']],pv],'prefix_only':pv}
  curves=targetmodel.mean_curve(future);rising=np.flatnonzero(curves[:,-1]>curves[:,0])
  gr=np.isin(gv,rising).astype(int);gr[gv<0]=-1;out={}
  for key in NAMES:
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=seed))
   model.fit(bx[key],gt);out[key]=model.predict_proba(bz[key]);fits+=1
   assert np.array_equal(model[-1].classes_,[0,1,2])
  audits.append(dict(seed=seed,prefix_fit_n=len(tr),future_fit_n=len(et),test_n=len(te),test_evaluable_n=int((gv>=0).sum()),training_class_counts=np.bincount(gt,minlength=3).tolist(),rising_classes_1based=(rising+1).tolist(),future_net_change=(curves[:,-1]-curves[:,0]).tolist(),future_grid=future,future_curves=curves.tolist(),future_model=targetmodel.to_dict(),prefix_model=prefmodel.to_dict()))
  return gv,gr,out,{k:out[k][:,rising].sum(axis=1) for k in NAMES}

 for fold in range(1,6):
  tr=np.flatnonzero(risk.fold.to_numpy()!=fold);te=np.flatnonzero(risk.fold.to_numpy()==fold)
  ip={k:np.full((len(tr),3),np.nan) for k in NAMES};ir={k:np.full(len(tr),np.nan) for k in NAMES};iy=np.full(len(tr),-1);iry=np.full(len(tr),-1)
  for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
   gv,gr,p,rp=one(tr[a],tr[b],SEED+t*1000+fold*100+j);iy[b]=gv;iry[b]=gr
   for k in NAMES:ip[k][b]=p[k];ir[k][b]=rp[k]
   print(tag,'outer',fold,'inner',j,'elapsed',round(time.time()-start),flush=True)
  gv,gr,p,rp=one(tr,te,SEED+t*1000+fold*100);truth[te]=gv;risingtruth[te]=gr
  ok=iy>=0;rok=iry>=0
  for k in NAMES:
   prob[k][te]=p[k];riseprob[k][te]=rp[k]
   for g in range(3):
    yy=(iy[ok]==g).astype(int);ff,tt,th=roc_curve(yy,ip[k][ok,g]);obj=tt-ff;obj[~np.isfinite(th)]=-np.inf;cut=float(th[np.argmax(obj)])
    alarms[k][te,g]=p[k][:,g]>=cut;cuts.append(dict(origin_h=t,mode=mode,fold=fold,model=k,target_class=g+1,threshold=cut))
   if len(np.unique(iry[rok]))==2:
    ff,tt,th=roc_curve(iry[rok],ir[k][rok]);obj=tt-ff;obj[~np.isfinite(th)]=-np.inf;cut=float(th[np.argmax(obj)]);risealarm[k][te]=rp[k]>=cut
  (R/(tag+'_fit_audit.json')).write_text(json.dumps(dict(origin_h=t,mode=mode,completed_outer_folds=fold,classifier_fits=fits,gbtm_fits=2*len(audits),records=audits),indent=2))
  print(tag,'completed outer',fold,'class counts',np.bincount(gv[gv>=0],minlength=3),'elapsed',round(time.time()-start),flush=True)
 out=risk[['subject_id','fold','recognition_hour','exit_h','death_exit']].copy();out['target']=truth;out['rising_target']=risingtruth;out['future_evaluable']=assess
 for k in NAMES:
  for g in range(3):out[f'{k}_p{g+1}']=prob[k][:,g];out[f'{k}_alarm{g+1}']=alarms[k][:,g]
  out[k+'_rising_p']=riseprob[k];out[k+'_rising_alarm']=risealarm[k]
 out.to_pickle(P/(tag+'_oof.pkl'));pd.DataFrame(cuts).to_csv(R/(tag+'_thresholds.csv'),index=False)
 assert np.array_equal(truth>=0,assess) and all(np.isfinite(x).all() for x in prob.values())
 metrics=[];tests=[];multiclass=[];rng=np.random.default_rng(SEED+t);yy=truth[assess];nn=len(yy)
 funcs={(k,g):WeightedMetrics((yy==g).astype(int),prob[k][assess,g]) for k in NAMES for g in range(3)}
 idx=[np.flatnonzero(yy==g) for g in range(3)];vals={key:[] for key in funcs};deltas={(k,g):[] for k in ['prefix_added','sofa_slope'] for g in range(3)}
 for b in range(10000):
  ids=np.concatenate([rng.choice(z,len(z),True) for z in idx]);weights=np.bincount(ids,minlength=nn).astype(float)
  active=list(funcs) if b<1000 else [(k,g) for k in ['latest_sofa','prefix_added','sofa_slope'] for g in range(3)]
  v={key:funcs[key](weights) for key in active}
  if b<1000:
   for key in funcs:vals[key].append(v[key])
  for key in deltas:deltas[key].append(v[key][0]-v['latest_sofa',key[1]][0])
 for (k,g),fn in funcs.items():
  point=fn(np.ones(nn));ci=np.quantile(vals[k,g],[.025,.975],axis=0);tn,fp,fn_,tp=confusion_matrix(yy==g,alarms[k][assess,g]).ravel()
  row=dict(origin_h=t,mode=mode,model=k,target_class=g+1,n=nn,positive_n=int((yy==g).sum()),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn_),sensitivity=tp/(tp+fn_),specificity=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else 0)
  for j,name in enumerate(['auroc','auprc','brier']):row[name]=point[j];row[name+'_lo']=ci[0,j];row[name+'_hi']=ci[1,j]
  metrics.append(row)
 for (k,g),dif in deltas.items():
  point=funcs[k,g](np.ones(nn))[0]-funcs['latest_sofa',g](np.ones(nn))[0];se=float(np.std(dif,ddof=1));lo,hi=np.quantile(dif,[.025,.975])
  tests.append(dict(origin_h=t,mode=mode,addition=k,reference='latest_sofa',target_class=g+1,delta_auroc=point,ci_low=lo,ci_high=hi,bootstrap_se=se,p_raw=float(2*norm.sf(abs(point/se))) if se else float(point==0)))
 for k in NAMES:
  mm=np.mean([np.asarray(vals[k,g])[:,0] for g in range(3)],axis=0);lo,hi=np.quantile(mm,[.025,.975]);point=np.mean([funcs[k,g](np.ones(nn))[0] for g in range(3)])
  multiclass.append(dict(origin_h=t,mode=mode,model=k,n=nn,macro_ovr_auroc=point,macro_lo=lo,macro_hi=hi,accuracy=accuracy_score(yy,prob[k][assess].argmax(axis=1)),log_loss=log_loss(yy,prob[k][assess],labels=[0,1,2]),confusion_matrix=json.dumps(confusion_matrix(yy,prob[k][assess].argmax(axis=1)).tolist())))
 pd.DataFrame(metrics).to_csv(R/(tag+'_metrics.csv'),index=False);pd.DataFrame(tests).to_csv(R/(tag+'_tests.csv'),index=False);pd.DataFrame(multiclass).to_csv(R/(tag+'_multiclass.csv'),index=False)
 (R/(tag+'_rising_status.json')).write_text(json.dumps(dict(partitions=len(audits),partitions_with_rising_class=sum(bool(a['rising_classes_1based']) for a in audits),all_partitions_have_rising_class=all(bool(a['rising_classes_1based']) for a in audits),oof_positive_n=int((risingtruth==1).sum()),oof_negative_n=int((risingtruth==0).sum()),status='Do not relabel the highest future SOFA class as deterioration. Inspect all recorded class curves; absence of a rising class is a model finding.'),indent=2))
 print(tag,'DONE',round(time.time()-start),'seconds',flush=True);print(pd.DataFrame(multiclass).to_string(index=False),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--time',type=int,choices=[12,24],required=True);a.add_argument('--mode',choices=['main','separated'],default='main');args=a.parse_args();main(args.time,args.mode)
