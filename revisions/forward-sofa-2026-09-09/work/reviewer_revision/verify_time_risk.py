from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from scipy.stats import norm
W=Path('work/multicohort');P=W/'private';R=W/'results';c=pd.read_pickle(P/'time_risk_cohort.pkl');m=pd.read_csv(R/'time_risk_metrics.csv');flow=pd.read_csv(R/'time_risk_flow.csv');leads=pd.read_csv(R/'time_risk_lead_intervals.csv')
assert len(c)==4856 and c.subject_id.is_unique and c.fold.nunique()==5
for r in m.itertuples():
 p=pd.read_pickle(P/f'time_risk_{r.target}_oof_{r.hour}h.pkl')
 assert len(p)==r.n and int(p.target.sum())==r.events and (p.fold==c.loc[p.index,'fold']).all()
 assert c.loc[p.index,'recognition_hour'].le(r.hour).all() and p.exit_hour.gt(r.hour).all()
 if r.target=='support':
  assert (~p.event_hour.le(r.hour)).all();assert np.array_equal(p.target.to_numpy(),p.event_hour.le(r.hour+12).astype(int).to_numpy())
 else:assert np.array_equal(p.target.to_numpy(),(p.death_exit&p.exit_hour.le(r.hour+12)).astype(int).to_numpy())
 for name,fn in [('auroc',roc_auc_score),('auprc',average_precision_score),('brier',brier_score_loss)]:assert abs(fn(p.target,p[r.model])-getattr(r,name))<1e-12
 tn,fp,fn,tp=confusion_matrix(p.target,p[r.model+'_alarm']).ravel();assert (tn,fp,fn,tp)==(r.tn,r.fp,r.fn,r.tp)
 if r.target=='support':
  d=p.loc[p.target.eq(1)&p[r.model+'_alarm'],'event_hour']-r.hour;q=leads.loc[leads.hour.eq(r.hour)&leads.model.eq(r.model)].iloc[0]
  assert d.gt(0).all() and d.le(12).all();np.testing.assert_allclose(d.quantile([.25,.5,.75]),q[['q1_h','median_h','q3_h']].to_numpy(float))
for r in flow.itertuples():
 assert r.source_n==r.dead_by_time+r.other_exit_by_time+r.present_not_yet_recognized+r.recognized_present_n
 assert r.recognized_present_n==r.prior_support_event_present_n+r.support_risk_n
 assert r.support_risk_n==r.support_events+r.nonevent_death_within12+r.nonevent_other_exit_within12+r.nonevent_observed_beyond12
checks=[]
for sid,r in c.loc[c.event_hour.le(60)].iterrows():
 eligible=[t for t in [2,4,6,8,12,24,48] if r.recognition_hour<=t<r.event_hour and t<r.exit_hour]
 eligible12=[t for t in eligible if r.event_hour<=t+12]
 checks.append(dict(any_prior_prediction=bool(eligible),any_12h_opportunity=bool(eligible12)))
z=pd.DataFrame(checks);op=dict(events_by60=len(z),with_any_prior_evaluated_origin=int(z.any_prior_prediction.sum()),without_any_prior_evaluated_origin=int((~z.any_prior_prediction).sum()),with_12h_prediction_opportunity=int(z.any_12h_opportunity.sum()),without_12h_prediction_opportunity=int((~z.any_12h_opportunity).sum()))
(R/'time_risk_prediction_opportunities.json').write_text(json.dumps(op,indent=2))
t=pd.read_csv(R/'time_risk_slope_tests.csv',float_precision='round_trip');np.testing.assert_allclose(t.p_raw,2*norm.sf(abs(t.difference/t.bootstrap_se)),rtol=1e-11)
# Independently verify the returned Holm family.
a=t.p_raw.to_numpy();order=np.argsort(a);adj=np.empty(len(a));adj[order]=np.minimum(1,np.maximum.accumulate(a[order]*np.arange(len(a),0,-1)));np.testing.assert_allclose(t.p_holm,adj,rtol=1e-12)
q=dict(unique_source_patients=4856,verified_oof_configurations=len(m),landmarks=flow.hour.tolist(),fixed_patient_folds_across_times=True,risk_population_partitions_verified=True,metric_counts_and_lead_quantiles_verified=True,paired_p_and_holm_verified=True,prediction_opportunities=op)
Path('work/reviewer_revision/time_risk_qa.json').write_text(json.dumps(q,indent=2));print(q)
