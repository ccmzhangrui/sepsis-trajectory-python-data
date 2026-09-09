from pathlib import Path
import subprocess,sys,json,os
import pandas as pd
W=Path('work/multicohort/mimiciii');O=Path('work/mimiciii')
subprocess.run([sys.executable,str(W/'04_fit_trajectories.py')],check=True)
y=pd.read_pickle(W/'private/trajectory_matrix.pkl');f=pd.read_pickle(O/'private/features_24h.pkl').loc[y.index];f.to_pickle(W/'private/features_24h.pkl')
a=json.loads((O/'feature_audit.json').read_text());a['n']=len(f);a['clinical_exit_filter']='earliest ICU or hospital discharge or recorded hospital death';(W/'feature_audit.json').write_text(json.dumps(a,indent=2))
for name in ['06_crossvalidate.py','07_pairwise.py']:
 subprocess.run([sys.executable,str(W/name)],check=True)
print('MIMIC-III final pipeline completed',flush=True)
