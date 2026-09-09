"""Executed time-specific SOFA-independent event validation; see time_risk_protocol.md."""
from pathlib import Path
import sys,os,json,hashlib
import numpy as np,pandas as pd,duckdb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve,confusion_matrix
from scipy.stats import norm
sys.path.insert(0,str(Path('work/multicohort').resolve()))
from figure_inference import WeightedMetrics,holm
from landmark_corrected_analysis import build_vital_hourly,statistics
W=Path('work/multicohort');P=W/'private';R=W/'results';M=W/'mimiciii/private'
ROOT=Path(os.environ['MIMICIII_DIR'])
SEED=20260916;TIMES=[2,4,6,8,12,24,48];PREFIX='time_risk_'

def write(data,name):
 if isinstance(data,pd.DataFrame):data.to_csv(R/(PREFIX+name+'.csv'),index=False)
 else:(R/(PREFIX+name+'.json')).write_text(json.dumps(data,indent=2))

def prepare():
 c=pd.read_pickle(M/'cohort.pkl').set_index('icustay_id').sort_values('subject_id')
 c['exit_hour']=(c.outtime-c.intime).dt.total_seconds()/3600;c['death_exit']=c.outtime.eq(c.deathtime)
 assert len(c)==4856 and c.subject_id.is_unique
 records={i:[] for i in c.index};raws=[];manifest=[]
 for drug in ['ventilation','norepinephrine','epinephrine','dopamine','phenylephrine','vasopressin']:
  p=ROOT/(drug+'_durations.csv');g=pd.read_csv(p,parse_dates=['starttime','endtime'])
  manifest.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(g)))
  g=g.merge(c[['intime','exit_hour']],left_on='icustay_id',right_index=True,validate='many_to_one')
  g['a']=(g.starttime-g.intime).dt.total_seconds()/3600;g['b']=np.minimum((g.endtime-g.intime).dt.total_seconds()/3600,g.exit_hour)
  g['kind']='ventilation' if drug=='ventilation' else 'vasopressor';raws.append(g[['icustay_id','a','b','kind']])
 for (sid,kind),g in pd.concat(raws).groupby(['icustay_id','kind']):
  intervals=[]
  for a,b in sorted(zip(g.a,g.b)):
   if not np.isfinite(a+b) or b<=a:continue
   if intervals and a<=intervals[-1][1]:intervals[-1][1]=max(b,intervals[-1][1])
   else:intervals.append([a,b])
  previous_end=-np.inf
  for a,b in intervals:
   if a>c.loc[sid,'recognition_hour'] and a<c.loc[sid,'exit_hour'] and previous_end<=max(0,a-6):records[sid].append((a,kind,b))
   previous_end=b
 c['event_hour']=[min(records[i])[0] if records[i] else np.nan for i in c.index]
 c['event_type']=[min(records[i])[1] if records[i] else 'no observed event' for i in c.index]
 c['fold']=-1
 for fold,(_,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(c),1):c.iloc[te,c.columns.get_loc('fold')]=fold
 c.to_pickle(P/(PREFIX+'cohort.pkl'))
 db=duckdb.connect('work/mimiciii/private/analysis.duckdb',read_only=True)
 census=db.execute('''SELECT f.subject_id,f.icustay_id,f.intime,least(f.outtime,a.dischtime,f.deathtime) AS clinical_exit,
 f.deathtime FROM mimiciii_derived.first_adult_mv f JOIN mimiciii.admissions a USING(hadm_id)''').fetchdf();db.close()
 census=census.loc[census.clinical_exit.gt(census.intime)].copy();exit_h=(census.clinical_exit-census.intime).dt.total_seconds()/3600;death=census.deathtime.eq(census.clinical_exit)
 cs=[]
 for t in TIMES:
  cs.append(dict(hour=t,first_adult_mv_n=len(census),dead_by_time=int((death&exit_h.le(t)).sum()),other_exit_by_time=int((~death&exit_h.le(t)).sum()),under_observation_n=int(exit_h.gt(t).sum())))
 write(pd.DataFrame(cs),'all_icu_census')
 write(dict(source_n=len(c),first_adult_mv_n=len(census),event_definition='First post-recognition recorded new ventilation/any-vasopressor episode after up to 6h of category absence; positive observed duration; no SOFA condition.',source_files=manifest,protocol_sha256=hashlib.sha256((W/'time_risk_protocol.md').read_bytes()).hexdigest(),event_n=int(c.event_hour.notna().sum()),event_by60_n=int(c.event_hour.le(60).sum()),features='Raw chart and store cutoff precedes aggregation; lab release times unavailable.'),'cohort_audit')
 return c

def features(t,c,v,s):
 f=pd.DataFrame(index=c.index);f['age']=c.admission_age.clip(upper=90);f['female']=c.gender.eq('F').astype(int)
 vv=v.loc[v.landmark_h.eq(t)].sort_values(['icustay_id','hour'])
 for var in ['hr','rr','temp','spo2']:
  vals={i:statistics(g.value.to_numpy(),g.hour.to_numpy()) for i,g in vv.loc[vv.variable.eq(var)].groupby('icustay_id')}
  d=pd.DataFrame.from_dict(vals,orient='index').reindex(c.index)
  for name in ['last','mean','sd','slope','n_hours']:f[var+'_'+name]=d[name]
  f[var+'_n_hours']=f[var+'_n_hours'].fillna(0)
 ss=s.loc[s.hour.le(t)&s.observed_components.ge(4)].sort_values(['icustay_id','hour'])
 vals={i:statistics(g.sofa_total.to_numpy(),g.hour.to_numpy()) for i,g in ss.groupby('icustay_id')}
 d=pd.DataFrame.from_dict(vals,orient='index').reindex(c.index);f['sofa_last']=d['last'];f['sofa_slope']=d['slope']
 return f

def fit(t,c,f,target,cols,target_name):
 y=np.asarray(target,dtype=int);pred={k:np.full(len(c),np.nan) for k in cols};alarm={k:np.zeros(len(c),bool) for k in cols};cuts=[]
 assert y.sum()>=5,(t,target_name,int(y.sum()))
 for fold in range(1,6):
  tr=np.flatnonzero(c.fold.to_numpy()!=fold);te=np.flatnonzero(c.fold.to_numpy()==fold)
  assert min(np.bincount(y[tr],minlength=2))>=3
  def one(itr,ite,seed):
   out={}
   for k,names in cols.items():
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=seed))
    model.fit(f.iloc[itr][names],y[itr]);out[k]=model.predict_proba(f.iloc[ite][names])[:,1]
   return out
  ip={k:np.full(len(tr),np.nan) for k in cols}
  for j,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED+fold).split(tr,y[tr]),1):
   result=one(tr[a],tr[b],SEED+100*fold+j)
   for k in cols:ip[k][b]=result[k]
  result=one(tr,te,SEED+100*fold)
  for k in cols:
   fpr,tpr,th=roc_curve(y[tr],ip[k]);obj=tpr-fpr;obj[~np.isfinite(th)]=-np.inf;cut=float(th[np.argmax(obj)])
   pred[k][te]=result[k];alarm[k][te]=result[k]>=cut;cuts.append(dict(hour=t,target=target_name,model=k,fold=fold,threshold=cut,n_train=len(tr),n_test=len(te),train_events=int(y[tr].sum())))
 out=c[['subject_id','fold','exit_hour','death_exit','event_hour']].copy();out['target']=y
 for k in cols:out[k]=pred[k];out[k+'_alarm']=alarm[k]
 out.to_pickle(P/f'{PREFIX}{target_name}_oof_{t}h.pkl')
 write(pd.DataFrame(cuts),f'{target_name}_thresholds_{t}h')
 print(t,target_name,'n',len(c),'events',int(y.sum()),'fits',20*len(cols),flush=True)
 return out

def summarize(t,p,keys,target,paired=False):
 y=p.target.to_numpy();n=len(p);func={k:WeightedMetrics(y,p[k]) for k in keys};rng=np.random.default_rng(SEED+t);idx=[np.flatnonzero(y==z) for z in [0,1]]
 vals={k:[] for k in keys};delta=[]
 for rep in range(10000 if paired else 1000):
  ix=np.r_[rng.choice(idx[0],len(idx[0]),True),rng.choice(idx[1],len(idx[1]),True)];w=np.bincount(ix,minlength=n).astype(float)
  active=keys if rep<1000 else ['sofa_level','sofa_slope'];v={k:func[k](w) for k in active}
  if rep<1000:
   for k in keys:vals[k].append(v[k])
  if paired:delta.append(v['sofa_slope'][0]-v['sofa_level'][0])
 rows=[];lead=[]
 for k in keys:
  point=func[k](np.ones(n));ci=np.quantile(vals[k],[.025,.975],axis=0);tn,fp,fn,tp=confusion_matrix(y,p[k+'_alarm']).ravel()
  row=dict(hour=t,target=target,model=k,n=n,events=int(y.sum()),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else 0,npv=tn/(tn+fn),fp_per100=100*fp/n)
  for j,name in enumerate(['auroc','auprc','brier']):row[name]=point[j];row[name+'_lo']=ci[0,j];row[name+'_hi']=ci[1,j]
  rows.append(row)
  if target=='support':
   hours=p.loc[p.target.eq(1)&p[k+'_alarm'],'event_hour'].to_numpy()-t
   q=np.quantile(hours,[.25,.5,.75]) if len(hours) else [None]*3
   lead.append(dict(hour=t,model=k,events=int(y.sum()),tp=int(tp),fp=int(fp),q1_h=q[0],median_h=q[1],q3_h=q[2],detected_at_least6h_n=int((hours>=6).sum())))
 test=None
 if paired:
  diff=func['sofa_slope'](np.ones(n))[0]-func['sofa_level'](np.ones(n))[0];sd=float(np.std(delta,ddof=1));lo,hi=np.quantile(delta,[.025,.975]);pv=float(2*norm.sf(abs(diff/sd))) if sd else (1. if diff==0 else 0.)
  test=dict(hour=t,n=n,events=int(y.sum()),difference=diff,ci_low=lo,ci_high=hi,bootstrap_se=sd,p_raw=pv,bootstrap_repeats=10000)
 return rows,lead,test

def main():
 c=prepare();s=pd.read_pickle(M/'sofa_hourly.pkl')
 print('extracting raw-cutoff vital histories',flush=True)
 # Same audited cutoff-first extractor, including the added 6-hour origin.
 v=build_vital_hourly(c.index);v.to_pickle(P/(PREFIX+'vitals.pkl'))
 latest=['age','female']+[z+'_last' for z in ['hr','rr','temp','spo2']]
 history=['age','female']+[z+'_'+a for z in ['hr','rr','temp','spo2'] for a in ['last','mean','sd','slope','n_hours']]
 cols=dict(latest=latest,history=history,sofa_level=history+['sofa_last'],sofa_slope=history+['sofa_last','sofa_slope'])
 allmetrics=[];alllead=[];tests=[];flows=[]
 for t in TIMES:
  ex=c.exit_hour.le(t);ready=c.recognition_hour.le(t)&~ex;prior=c.event_hour.le(t);risk=c.loc[ready&~prior].copy();all_alive=c.loc[ready].copy()
  y=risk.event_hour.le(t+12).astype(int);non=risk.loc[y.eq(0)]
  flows.append(dict(hour=t,source_n=len(c),dead_by_time=int((ex&c.death_exit).sum()),other_exit_by_time=int((ex&~c.death_exit).sum()),present_not_yet_recognized=int((~ex&~c.recognition_hour.le(t)).sum()),recognized_present_n=int(ready.sum()),prior_support_event_present_n=int((ready&prior).sum()),support_risk_n=len(risk),support_events=int(y.sum()),nonevent_death_within12=int((non.death_exit&non.exit_hour.le(t+12)).sum()),nonevent_other_exit_within12=int((~non.death_exit&non.exit_hour.le(t+12)).sum()),nonevent_observed_beyond12=int(non.exit_hour.gt(t+12).sum())))
  f=features(t,c,v,s);f.loc[risk.index].to_pickle(P/f'{PREFIX}support_features_{t}h.pkl')
  p=fit(t,risk,f.loc[risk.index],y,cols,'support');r,l,q=summarize(t,p,list(cols),'support',True);allmetrics+=r;alllead+=l;tests.append(q)
  ym=(all_alive.death_exit&all_alive.exit_hour.le(t+12)).astype(int)
  p=fit(t,all_alive,f.loc[all_alive.index],ym,{'history':history},'death');r,_,_=summarize(t,p,['history'],'death');allmetrics+=r
  write(pd.DataFrame(allmetrics),'metrics');write(pd.DataFrame(flows),'flow');write(pd.DataFrame(alllead),'lead_intervals')
 pdtests=pd.DataFrame(tests);pdtests['p_holm']=holm(pdtests.p_raw.to_numpy());write(pdtests,'slope_tests')
 write(dict(seed=SEED,landmarks=TIMES,patient_outer_folds=5,inner_folds=3,classifier_fits=len(TIMES)*20*5,outcome_window_hours=12,predictor_columns=cols,death_predictors=history,description='Fixed patient folds shared across times; risk eligibility assessed separately at each prediction time; no 24-hour survivor selection.'),'model_audit')
 print(pd.DataFrame(flows).to_string(index=False),flush=True)
 print(pd.DataFrame(allmetrics)[['hour','target','model','n','events','auroc','auprc','ppv','tp','fp']].to_string(index=False),flush=True)

if __name__=='__main__':main()
