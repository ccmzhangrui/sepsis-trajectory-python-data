from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
W=Path('work/mimiciii');p=pd.read_pickle(W/'private'/'trajectory_oof_predictions.pkl');r=np.random.default_rng(20260908)
pairs=[('truncated_posterior_24h','sofa_without_posterior'),('truncated_posterior_12h','sofa_without_posterior'),('truncated_posterior_24h','no_sofa_inputs')]
ids=[np.flatnonzero(p.target.eq(j)) for j in [0,1]];values={pair:[] for pair in pairs}
for _ in range(1000):
 ix=np.r_[r.choice(ids[0],len(ids[0]),replace=True),r.choice(ids[1],len(ids[1]),replace=True)];b=p.iloc[ix]
 for a,z in pairs:values[(a,z)].append(roc_auc_score(b.target,b[a])-roc_auc_score(b.target,b[z]))
rows=[]
for a,z in pairs:
 lo,hi=np.quantile(values[(a,z)],[.025,.975]);rows.append({'model_a':a,'model_b':z,'auroc_difference':roc_auc_score(p.target,p[a])-roc_auc_score(p.target,p[z]),'ci_low':lo,'ci_high':hi,'bootstrap_repeats':1000})
pd.DataFrame(rows).to_csv(W/'results'/'paired_auroc_differences.csv',index=False);print(pd.DataFrame(rows).to_json(orient='records'))
