from pathlib import Path
import json,os,time
import numpy as np,pandas as pd
from gbtm import CensoredGBTM
from scipy.optimize._numdiff import approx_derivative

W=Path('work/mimiciii');P=W/'private';R=W/'results';R.mkdir(exist_ok=True)
cohort=pd.read_pickle(P/'cohort.pkl');sofa=pd.read_pickle(P/'sofa_hourly.pkl')
times=np.arange(6,61,6)
cohort=cohort.loc[cohort.recognition_hour.le(24)&(cohort.outtime-cohort.intime).dt.total_seconds().gt(86400)]
s=sofa.loc[sofa.icustay_id.isin(cohort.icustay_id)&sofa.hour.isin(times)]
y=s.pivot(index='icustay_id',columns='hour',values='sofa_total').reindex(columns=times)
obs=s.pivot(index='icustay_id',columns='hour',values='observed_components').reindex(columns=times)
y=y.where(obs.ge(4));y=y.loc[y.loc[:,times<=24].notna().sum(axis=1).ge(3)]
assert not y.index.duplicated().any()
y.to_pickle(P/'trajectory_matrix.pkl');os.chmod(P/'trajectory_matrix.pkl',0o600)
# A meaningful analytic-gradient check includes both censoring boundaries and missing data.
test_y=np.array([[0.,2.,5.,24.,np.nan],[1.,0.,24.,6.,4.]])
test_t=np.array([6,12,24,48,60]);x=np.vander(test_t/60,3,increasing=True)
def obj(theta):
    lp,gm,gs=CensoredGBTM._terms(test_y,x@theta[:-1],np.exp(theta[-1]))
    return -lp.sum(),np.r_[-x.T@gm.sum(axis=0),-gs.sum()]
theta=np.array([2.,3.,-.2,np.log(2.)]);numeric=approx_derivative(lambda z:obj(z)[0],theta).ravel()
assert np.allclose(numeric,obj(theta)[1],rtol=1e-5,atol=1e-5)
fits=[];summaries=[]
for k in [2,3,4,5]:
    print(json.dumps({'start_k':k,'n':len(y)}),flush=True);t=time.time()
    model=CensoredGBTM(k=k,starts=5,max_iter=150).fit(y.to_numpy(),times)
    post=model.predict_proba(y.to_numpy(),times);label=post.argmax(axis=1);curve=model.mean_curve(times)
    assert np.allclose(post.sum(axis=1),1) and model.best_start['monotone_up_to_tolerance']
    fits.append(model.to_dict());(R/f'gbtm_k{k}.json').write_text(json.dumps(model.to_dict(),indent=2))
    group_rows=[]
    for g in range(k):
        select=label==g;group=y.iloc[np.flatnonzero(select)]
        row={'k':k,'class':g+1,'n':int(select.sum()),'proportion':float(select.mean()),'mean_assignment_posterior':float(post[select,g].mean()),'model_sofa_6h':float(curve[g,0]),'model_sofa_60h':float(curve[g,-1]),'model_change_6_to_60':float(curve[g,-1]-curve[g,0])}
        group_rows.append(row)
        for index,h in enumerate(times):
            observed=group[h].dropna()
            summaries.append({'k':k,'class':g+1,'hour':int(h),'n_observed':len(observed),'observed_mean':float(observed.mean()),'observed_sem':float(observed.sem()),'model_mean':float(curve[g,index])})
    pd.DataFrame(group_rows).to_csv(R/f'class_summary_k{k}.csv',index=False)
    np.savez_compressed(P/f'posterior_k{k}.npz',icustay_id=y.index.to_numpy(),posterior=post)
    print(json.dumps({'k':k,'bic':model.bic,'seconds':round(time.time()-t,1),'classes':group_rows}),flush=True)
pd.DataFrame(summaries).to_csv(R/'trajectory_curves.csv',index=False)
fit_table=[{'k':m['k'],'n':m['n'],'parameters':m['n_parameters'],'log_likelihood':m['loglik'],'bic':m['bic'],'converged':m['best_start']['em_converged'],'best_start':m['best_start']['start']} for m in fits]
pd.DataFrame(fit_table).to_csv(R/'model_selection.csv',index=False)
metadata={'cohort_known_at_24h_n':len(cohort),'minimum_three_measured_sofa_times_by_24h_n':len(y),'timepoints_hours':times.tolist(),'missing_rule':'At least four observed organ systems per SOFA time point; fewer than four -> missing; no post-exit fill','gbtm':'quadratic censored normal mixture with class-specific residual SD and five starts','best_k_by_bic':int(min(fits,key=lambda f:f['bic'])['k']),'three_classes':'prespecified comparison only; no forcing the BIC minimum','gradient_check_max_abs_error':float(abs(numeric-obj(theta)[1]).max()),'classification_and_warning_metrics':'Not produced by this descriptive fit.'}
(R/'trajectory_analysis_metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
