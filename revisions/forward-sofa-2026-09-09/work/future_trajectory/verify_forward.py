from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix,accuracy_score,log_loss
from scipy.stats import norm
sys.path.insert(0,str(Path('work/multicohort/mimiciii').resolve()))
from gbtm import CensoredGBTM
sys.path.insert(0,str(Path('work/multicohort').resolve()))
from figure_inference import holm
W=Path('work/future_trajectory');R=W/'results';P=W/'private'
mm=[];tt=[];multi=[];flows=[];curve_rows=[];class_counts=[]
for mode in ['main','separated']:
 for t in [12,24]:
  tag=f'{mode}_{t}h';p=pd.read_pickle(P/(tag+'_oof.pkl'));Y=pd.read_pickle(P/(tag+'_future_scores.pkl'))
  f=json.loads((R/(tag+'_flow.json')).read_text());a=json.loads((R/(tag+'_fit_audit.json')).read_text());m=pd.read_csv(R/(tag+'_metrics.csv'));tests=pd.read_csv(R/(tag+'_tests.csv'),float_precision='round_trip');mac=pd.read_csv(R/(tag+'_multiclass.csv'))
  assert a['classifier_fits']==100 and a['gbtm_fits']==40 and a['completed_outer_folds']==5
  assert p.subject_id.is_unique and p.recognition_hour.le(t).all() and p.exit_h.gt(t).all()
  assert p.future_evaluable.equals(Y.notna().sum(axis=1).ge(3))
  assert f['source_n']==f['dead_by_origin']+f['other_exit_by_origin']+f['present_not_recognized']+f['recognized_present_n']
  assert f['recognized_present_n']==f['insufficient_prefix_n']+f['prefix_eligible_n']
  assert f['prefix_eligible_n']==f['future_evaluable_n']+f['future_unassessable_n']
  assert f['future_unassessable_n']==f['unassessable_death_by60']+f['unassessable_other_exit_by60']+f['unassessable_recording_gap']
  for fold,rec in enumerate(a['records'][3::4],1):
   z=rec['future_model'];g=CensoredGBTM();g.beta=np.array(z['beta']);g.sigma=np.array(z['sigma']);g.pi=np.array(z['pi'])
   q=p.loc[p.fold.eq(fold)&p.future_evaluable];truth=g.predict_proba(Y.loc[q.index].to_numpy(),Y.columns).argmax(axis=1)
   assert np.array_equal(truth,q.target)
   curves=g.mean_curve(Y.columns);rise=np.flatnonzero(curves[:,-1]>curves[:,0]);assert np.array_equal(np.isin(truth,rise).astype(int),q.rising_target)
   for cls in range(3):
    class_counts.append(dict(mode=mode,origin_h=t,outer_fold=fold,target_class=cls+1,test_evaluable_n=len(q),class_n=int((q.target==cls).sum()),first_future_sofa=curves[cls,0],last_future_sofa=curves[cls,-1],net_change=curves[cls,-1]-curves[cls,0],rising=bool(cls in rise)))
    for j,h in enumerate(Y.columns):curve_rows.append(dict(mode=mode,origin_h=t,outer_fold=fold,target_class=cls+1,hour=int(h),fitted_sofa=curves[cls,j]))
  q=p.loc[p.future_evaluable];assert len(q)==f['future_evaluable_n']
  for r in m.itertuples():
   yy=q.target.eq(r.target_class-1);score=q[f'{r.model}_p{r.target_class}'];alarm=q[f'{r.model}_alarm{r.target_class}']
   assert yy.sum()==r.positive_n
   for val,fn in [(r.auroc,roc_auc_score),(r.auprc,average_precision_score),(r.brier,brier_score_loss)]:assert abs(val-fn(yy,score))<1e-12
   assert tuple(confusion_matrix(yy,alarm).ravel())==(r.tn,r.fp,r.fn,r.tp)
  for r in mac.itertuples():
   score=q[[f'{r.model}_p{j}' for j in [1,2,3]]].to_numpy();np.testing.assert_allclose(score.sum(axis=1),1)
   assert abs(roc_auc_score(q.target,score,multi_class='ovr',average='macro')-r.macro_ovr_auroc)<1e-12
   assert abs(log_loss(q.target,score)-r.log_loss)<1e-12
  np.testing.assert_allclose(tests.p_raw,2*norm.sf(abs(tests.delta_auroc/tests.bootstrap_se)),rtol=1e-12)
  mm.append(m);tt.append(tests);multi.append(mac);flows.append(f)
mt=pd.concat(mm,ignore_index=True);tests=pd.concat(tt,ignore_index=True);mc=pd.concat(multi,ignore_index=True)
for mode in ['main','separated']:
 mask=tests['mode'].eq(mode);assert mask.sum()==12;tests.loc[mask,'p_holm']=holm(tests.loc[mask,'p_raw'].to_numpy())
for name,df in [('all_metrics',mt),('all_tests',tests),('all_multiclass',mc),('all_flow',pd.DataFrame(flows)),('outer_curves',pd.DataFrame(curve_rows)),('outer_classes',pd.DataFrame(class_counts))]:df.to_csv(R/(name+'.csv'),index=False)
q=dict(configurations=4,classifier_fits=400,gbtm_fits=160,model_configurations=20,class_specific_evaluations=60,source_population_partitions_verified=True,outer_labels_reproduced_from_saved_training_models=True,all_oof_point_metrics_independently_recalculated=True,confusion_counts_verified=True,paired_p_recalculated=True,holm_families={'main':12,'separated':12},interpretation='Future SOFA phenotype prediction. Rising class unstable; high future SOFA is not automatically a deterioration class.')
(W/'verification.json').write_text(json.dumps(q,indent=2));print(json.dumps(q,indent=2))
print(tests.loc[tests.target_class.eq(3)].to_string(index=False))
