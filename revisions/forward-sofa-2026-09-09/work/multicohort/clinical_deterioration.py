"""New operational organ-deterioration endpoint after ICU hour 24.

See deterioration_protocol_locked.md. This is a retrospective, separately defined
endpoint and one-landmark alert interval, not the original serial warning-time model.
"""
from pathlib import Path
import os, sys, json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
sys.path.insert(0,str(Path('work/multicohort/mimiciii').resolve()))
from gbtm import CensoredGBTM
from figure_inference import WeightedMetrics, holm
from scipy.stats import norm

W=Path('work/multicohort');P=W/'private';R=W/'results';M=W/'mimiciii/private'
ROOT=Path(os.environ['MIMICIII_DIR'])
SEED=20260913


def merged_intervals(g):
    out=[]
    for start,end in sorted(zip(g.start_hour,g.end_hour)):
        if not np.isfinite(start+end) or end<=start:continue
        if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
        else:out.append([start,end])
    return out


def extract():
    final_ids=pd.read_pickle(M/'trajectory_matrix.pkl').index
    c=pd.read_pickle(M/'cohort.pkl').set_index('icustay_id').loc[final_ids].copy()
    assert len(c)==3744 and c.subject_id.is_unique
    s=pd.read_pickle(M/'sofa_hourly.pkl')
    baseline=s.loc[s.hour.eq(24)&s.observed_components.ge(4)].set_index('icustay_id').sofa_total
    source_n=len(c);c=c.loc[c.index.intersection(baseline.index)].copy()
    c['sofa24']=baseline.reindex(c.index);c['exit_hour']=(c.outtime-c.intime).dt.total_seconds()/3600
    groups={i:g.sort_values('hour') for i,g in s.loc[s.observed_components.ge(4)&s.hour.gt(24)].groupby('icustay_id')}
    supports=[];manifest=[]
    for drug in ['ventilation','norepinephrine','epinephrine','dopamine','phenylephrine','vasopressin']:
        path=ROOT/(drug+'_durations.csv');raw=pd.read_csv(path,parse_dates=['starttime','endtime'])
        manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(raw)})
        raw=raw.merge(c[['intime','exit_hour']],left_on='icustay_id',right_index=True,validate='many_to_one')
        raw['start_hour']=(raw.starttime-raw.intime).dt.total_seconds()/3600
        raw['end_hour']=np.minimum((raw.endtime-raw.intime).dt.total_seconds()/3600,raw.exit_hour)
        raw['support']='ventilation' if drug=='ventilation' else 'vasopressor'
        supports.append(raw[['icustay_id','support','start_hour','end_hour']])
    support=pd.concat(supports,ignore_index=True)
    episodes={}
    for (sid,kind),g in support.groupby(['icustay_id','support']):
        intervals=merged_intervals(g)
        # Any use of this category during the six-hour washout makes it ineligible.
        if any(a<=24 and b>18 for a,b in intervals):continue
        episodes[(sid,kind)]=[(a,b) for a,b in intervals if 24<a<=60 and b-a>=1]
    records=[];candidates=[]
    for sid,row in c.iterrows():
        g=groups.get(sid,pd.DataFrame(columns=['hour','sofa_total']))
        event=[];prior_event=[];unknown=False;prior_unknown=False
        for kind in ['ventilation','vasopressor']:
            for onset,end in episodes.get((sid,kind),[]):
                win=g.loc[g.hour.between(onset-6,onset+6)]
                prior=g.loc[g.hour.between(onset-6,onset)]
                confirmed=len(win)>0 and bool((win.sofa_total>=row.sofa24+2).any())
                confirmed_prior=len(prior)>0 and bool((prior.sofa_total>=row.sofa24+2).any())
                if confirmed:event.append((onset,kind))
                if confirmed_prior:prior_event.append((onset,kind))
                unknown |= len(win)==0;prior_unknown |= len(prior)==0
                candidates.append(dict(icustay_id=sid,support=kind,start_hour=onset,end_hour=end,
                    baseline_sofa=row.sofa24,confirmation_n=len(win),prior_confirmation_n=len(prior),
                    max_sofa=float(win.sofa_total.max()) if len(win) else None,
                    confirmed=confirmed,confirmed_prior=confirmed_prior))
        event.sort();prior_event.sort()
        records.append(dict(icustay_id=sid,subject_id=row.subject_id,sofa24=row.sofa24,
            exit_hour=row.exit_hour,event=int(bool(event)) if event or not unknown else -1,
            event_hour=event[0][0] if event else np.nan,event_type=event[0][1] if event else '',
            prior_event=int(bool(prior_event)) if prior_event or not prior_unknown else -1,
            prior_event_hour=prior_event[0][0] if prior_event else np.nan,
            death_at_exit=bool(pd.notna(row.deathtime) and row.deathtime==row.outtime)))
    out=pd.DataFrame(records).set_index('icustay_id')
    all_classified=out.loc[out.event.ge(0)].copy()
    assert all_classified.subject_id.is_unique
    assert (all_classified.loc[all_classified.event.eq(1),'event_hour']>24).all()
    assert (all_classified.loc[all_classified.event.eq(1),'event_hour']<=60).all()
    assert (all_classified.loc[all_classified.event.eq(1),'event_hour']<all_classified.loc[all_classified.event.eq(1),'exit_hour']).all()
    all_classified.to_pickle(P/'deterioration_cohort.pkl')
    pd.DataFrame(candidates).to_pickle(P/'deterioration_episode_audit.pkl')
    neg=all_classified.loc[all_classified.event.eq(0)]
    audit=dict(source_n=source_n,evaluable_baseline_n=len(c),baseline_excluded=source_n-len(c),
        indeterminate_n=int(out.event.lt(0).sum()),n=len(all_classified),events=int(all_classified.event.sum()),
        first_event_types=all_classified.loc[all_classified.event.eq(1),'event_type'].value_counts().to_dict(),
        candidate_episodes=len(candidates),prior_definition_events=int(all_classified.prior_event.eq(1).sum()),
        prior_definition_indeterminate_n=int(all_classified.prior_event.lt(0).sum()),
        nonevent_death_before60=int((neg.death_at_exit&neg.exit_hour.le(60)).sum()),
        nonevent_other_exit_before60=int((~neg.death_at_exit&neg.exit_hour.le(60)).sum()),
        endpoint='New recorded ventilation or any-vasopressor support in (24,60] h, absent during (18,24] h, lasting >=1 h, and evaluable SOFA >=hour-24 value+2 within +/-6 h after the landmark.',
        event_clock='Support start. Later SOFA is used only to confirm outcomes, not shift onset or form predictors.',
        inference_scope='Retrospective operational proxy; no dose escalation, no adjudicated original-event replication; competing exits terminate observation.',
        source_files=manifest,protocol_sha256=hashlib.sha256((W/'deterioration_protocol_locked.md').read_bytes()).hexdigest())
    (R/'deterioration_cohort_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
    return all_classified


def threshold(y,p):
    fpr,tpr,t=roc_curve(y,p);q=tpr-fpr;q[~np.isfinite(t)]=-np.inf
    return float(t[np.argmax(q)])


def models(c):
    f=pd.read_pickle(M/'features_24h.pkl').loc[c.index]
    yy=pd.read_pickle(M/'trajectory_matrix.pkl').loc[c.index,[6,12,18,24]].to_numpy()
    y=c.event.to_numpy(dtype=int)
    history=['age_topcoded90','female']+[v+'_'+z for v in ['hr','rr','temp','spo2'] for z in ['last','mean','sd','slope','n_hours']]
    latest=['age_topcoded90','female']+[v+'_last' for v in ['hr','rr','temp','spo2']]
    sofa=[x for x in f if x.startswith('sofa_')]
    names={'latest':latest,'history':history,'sofa_history':history+sofa,'prefix_trajectory':history+sofa}
    predictions={k:np.full(len(c),np.nan) for k in names};alarms={k:np.zeros(len(c),bool) for k in names}
    folds=np.full(len(c),-1);fits=[];thresholds=[]
    def one(tr,te,seed):
        g=CensoredGBTM(k=3,degree=2,starts=3,max_iter=500,seed=seed).fit(yy[tr],[6,12,18,24])
        fits.append(g.to_dict())
        a=g.predict_proba(yy[tr],[6,12,18,24]);b=g.predict_proba(yy[te],[6,12,18,24]);out={}
        for k,cols in names.items():
            xt=f.iloc[tr][cols].to_numpy(dtype=float,na_value=np.nan)
            xv=f.iloc[te][cols].to_numpy(dtype=float,na_value=np.nan)
            if k=='prefix_trajectory':xt=np.c_[xt,a];xv=np.c_[xv,b]
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,
                min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=seed))
            model.fit(xt,y[tr]);out[k]=model.predict_proba(xv)[:,1]
        return out
    outer=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for fold,(tr,te) in enumerate(outer.split(f,y),1):
        print('deterioration outer fold',fold,'start',flush=True)
        inner=StratifiedKFold(3,shuffle=True,random_state=SEED+fold)
        ip={k:np.full(len(tr),np.nan) for k in names}
        for j,(itr,ite) in enumerate(inner.split(f.iloc[tr],y[tr]),1):
            result=one(tr[itr],tr[ite],SEED+100*fold+j)
            for k in names:ip[k][ite]=result[k]
        result=one(tr,te,SEED+100*fold)
        for k in names:
            cut=threshold(y[tr],ip[k]);predictions[k][te]=result[k];alarms[k][te]=result[k]>=cut
            thresholds.append(dict(fold=fold,model=k,threshold=cut,n_train=len(tr),n_test=len(te)))
        folds[te]=fold
        print('deterioration outer fold',fold,'complete',flush=True)
    assert all(np.isfinite(v).all() for v in predictions.values())
    p=c[['event','event_hour','prior_event','prior_event_hour']].copy();p['fold']=folds
    for k in names:p[k]=predictions[k];p[k+'_alarm']=alarms[k]
    p.to_pickle(P/'deterioration_oof.pkl');pd.DataFrame(thresholds).to_csv(R/'deterioration_thresholds.csv',index=False)
    (R/'deterioration_model_audit.json').write_text(json.dumps(dict(seed=SEED,n=len(c),events=int(y.sum()),
        gbtml_fits=len(fits),classifier_fits=4*len(fits),all_gbtm_converged=all(g['best_start']['em_converged'] for g in fits),
        predictor_registry=names,prefix_extra_predictors=3,gbtm_fits=fits,
        label='Fixed operational future event; identical target across folds and models',
        model_policy='500-iteration ceiling for GBTM; otherwise the established phenotype-classifier settings; no hyperparameter search'),indent=2))
    return p


def summaries(p):
    y=p.event.to_numpy();n=len(p);names=['latest','history','sofa_history','prefix_trajectory']
    funcs={k:WeightedMetrics(y,p[k]) for k in names};rng=np.random.default_rng(SEED+1)
    strata=[np.flatnonzero(y==v) for v in [0,1]];samples={k:[] for k in names}
    pairs=[('prefix_trajectory','sofa_history'),('sofa_history','history'),('history','latest')]
    deltas={pair:[] for pair in pairs}
    for rep in range(10000):
        ids=np.r_[rng.choice(strata[0],len(strata[0]),True),rng.choice(strata[1],len(strata[1]),True)]
        weights=np.bincount(ids,minlength=n).astype(float);vals={k:fn(weights) for k,fn in funcs.items()}
        for a,b in pairs:deltas[(a,b)].append(vals[a][0]-vals[b][0])
        if rep<1000:
            for k in names:samples[k].append(vals[k])
    rows=[];lead=[];sensitivity=[]
    for k in names:
        r={'model':k,'n':n,'events':int(y.sum())};actual=funcs[k](np.ones(n));ci=np.quantile(samples[k],[.025,.975],axis=0)
        for j,metric in enumerate(['auroc','auprc','brier']):r[metric]=actual[j];r[metric+'_lo']=ci[0,j];r[metric+'_hi']=ci[1,j]
        tn,fp,fn,tp=confusion_matrix(y,p[k+'_alarm']).ravel()
        r.update(tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),ppv=tp/(tp+fp),npv=tn/(tn+fn),false_positives_per100=100*fp/n)
        rows.append(r)
        hours=(p.loc[p.event.eq(1)&p[k+'_alarm'],'event_hour']-24).to_numpy()
        medci=np.quantile([np.median(rng.choice(hours,len(hours),True)) for _ in range(1000)],[.025,.975])
        q=np.quantile(hours,[.25,.5,.75])
        lead.append(dict(model=k,true_positive_n=len(hours),all_event_n=int(y.sum()),median_hours=q[1],q1_hours=q[0],q3_hours=q[2],median_ci_low=medci[0],median_ci_high=medci[1],
            detected_at_least6h_n=int((hours>=6).sum()),detected_at_least12h_n=int((hours>=12).sum()),
            detected_at_least6h_fraction=(hours>=6).sum()/y.sum(),detected_at_least12h_fraction=(hours>=12).sum()/y.sum(),
            scope='Recorded support onset minus single ICU-hour-24 prediction, among true-positive alerts; not serial first-threshold-crossing warning time.'))
        sub=p.loc[p.prior_event.ge(0)];truth=sub.prior_event.to_numpy();pred=sub[k]
        tn,fp,fn,tp=confusion_matrix(truth,sub[k+'_alarm']).ravel()
        sensitivity.append(dict(model=k,n=len(sub),events=int(truth.sum()),auroc=roc_auc_score(truth,pred),auprc=average_precision_score(truth,pred),brier=brier_score_loss(truth,pred),tp=int(tp),fp=int(fp),sensitivity=tp/(tp+fn),ppv=tp/(tp+fp),scope='Prior-only SOFA confirmation; fixed primary-label model predictions and thresholds, no refitting.'))
    paired=[]
    for a,b in pairs:
        d=funcs[a](np.ones(n))[0]-funcs[b](np.ones(n))[0];vals=np.array(deltas[(a,b)]);se=vals.std(ddof=1);lo,hi=np.quantile(vals,[.025,.975])
        paired.append(dict(model_a=a,model_b=b,metric='auroc',difference=d,ci_low=lo,ci_high=hi,bootstrap_se=se,z=d/se,p_raw=2*norm.sf(abs(d/se)),bootstrap_repeats=10000))
    paired=pd.DataFrame(paired);paired['p_holm']=holm(paired.p_raw)
    pd.DataFrame(rows).to_csv(R/'deterioration_metrics.csv',index=False)
    pd.DataFrame(lead).to_csv(R/'deterioration_lead_time.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(R/'deterioration_prior_confirmation_sensitivity.csv',index=False)
    paired.to_csv(R/'deterioration_paired_inference.csv',index=False)
    alltimes=p.loc[p.event.eq(1),'event_hour']-24
    (R/'deterioration_timing_summary.json').write_text(json.dumps({'all_event_n':len(alltimes),'all_event_hours_q1_median_q3':np.quantile(alltimes,[.25,.5,.75]).tolist(),'bootstrap_seed':SEED+1,'prediction_hours':24,'outcome_window_hours':[24,60]},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(lead).to_string(index=False),flush=True)
    print(paired.to_string(index=False),flush=True)


if __name__=='__main__':
    if '--summaries-only' in sys.argv:
        summaries(pd.read_pickle(P/'deterioration_oof.pkl'))
    else:
        c=extract();p=models(c);summaries(p)
