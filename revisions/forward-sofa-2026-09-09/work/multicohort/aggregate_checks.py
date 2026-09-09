from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,os
W=Path('work/multicohort');M=W/'mimiciii';R=W/'results';P=M/'private'
co=pd.read_pickle(P/'cohort.pkl').set_index('icustay_id');Y=pd.read_pickle(P/'trajectory_matrix.pkl');co=co.loc[Y.index]
rows=[]
for k in [3,5]:
 p=np.load(P/f'posterior_k{k}.npz');assert np.array_equal(p['icustay_id'],Y.index.to_numpy());labels=p['posterior'].argmax(axis=1)+1
 for g in range(1,k+1):
  d=co.hospital_expire_flag.to_numpy()[labels==g];n=len(d);events=int(d.sum());rate=events/n;z=1.95996398454;den=1+z*z/n;center=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
  rows.append({'k':k,'class':g,'n':n,'deaths':events,'mortality':rate,'wilson_lo':center-half,'wilson_hi':center+half,'scope':'Descriptive association with full6-60h phenotype; not prospective trajectory-based mortality prediction'})
pd.DataFrame(rows).to_csv(R/'mimiciii_mortality_by_trajectory.csv',index=False)
a=json.loads((M/'results/trajectory_analysis_metadata.json').read_text());c=json.loads((M/'results/crossvalidation_audit.json').read_text())
assert all(r['full_fit_converged'] and all(q['converged'] for q in r['prefix']) for r in c['fold_records'])
assert pd.read_csv(M/'results/model_selection.csv').converged.all()
source=Path(os.environ['MIMICIV_XLSX']);sha=hashlib.sha256(source.read_bytes()).hexdigest();iv=json.loads(Path('work/reanalysis/results/summary.json').read_text());assert iv['source_sha256']==sha
from sklearn.metrics import roc_auc_score
pred=np.load('work/reanalysis/results/local_predictions.npz');print('MIMICIV cached prediction keys',pred.files,flush=True)
qa={'mimiciii_n':len(co),'mimiciii_deaths':int(co.hospital_expire_flag.sum()),'hospital_death_or_discharge_before24_excluded':25,'all_descriptive_and_crossvalidation_trajectory_models_converged':True,'mimiciv_source_sha256_verified_current':True,'mimiciv_source_sha256':sha,'no_warning_time_estimate_used':True}
(R/'aggregate_qa.json').write_text(json.dumps(qa,indent=2));print(pd.DataFrame(rows).to_json(orient='records'),flush=True)
