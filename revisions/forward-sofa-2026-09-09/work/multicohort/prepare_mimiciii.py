from pathlib import Path
import json,os,shutil
import pandas as pd,duckdb
O=Path('work/mimiciii');W=Path('work/multicohort/mimiciii');P=W/'private';P.mkdir(parents=True,exist_ok=True);os.chmod(P,0o700);(W/'results').mkdir(exist_ok=True)
c=duckdb.connect(str(O/'private/analysis.duckdb'),read_only=True)
co=pd.read_pickle(O/'private/cohort.pkl');ad=c.execute('select hadm_id,dischtime from mimiciii.admissions').fetchdf();co=co.merge(ad,on='hadm_id',validate='many_to_one');co['raw_outtime']=co.outtime;co['outtime']=co[['outtime','dischtime','deathtime']].min(axis=1)
prior24=(co.recognition_hour<=24)&((co.raw_outtime-co.intime).dt.total_seconds()>86400)
keep=(co.recognition_time<=co.outtime)&(co.outtime>co.intime)
co=co.loc[keep].copy();s=pd.read_pickle(O/'private/sofa_hourly.pkl');s=s.merge(co[['icustay_id','outtime']],on='icustay_id',suffixes=('','_effective'),validate='many_to_one');excluded=int((s.endtime>s.outtime_effective).sum());s=s.loc[s.endtime<=s.outtime_effective].drop(columns='outtime_effective')
co.to_pickle(P/'cohort.pkl');s.to_pickle(P/'sofa_hourly.pkl')
for name in ['gbtm.py','04_fit_trajectories.py','06_crossvalidate.py','07_pairwise.py']:
 code=(O/name).read_text().replace("Path('work/mimiciii')","Path('work/multicohort/mimiciii')")
 if name=='04_fit_trajectories.py':code=code.replace('max_iter=150','max_iter=500')
 (W/name).write_text(code)
a={'original_trajectory_n':3769,'death_or_hospital_discharge_by24_exclusions_in_original_trajectory_set':25,'post_clinical_exit_sofa_rows_removed':excluded,'exit_rule':'earliest ICU discharge, hospital discharge, recorded hospital death; no score after clinical exit','qualification_at_or_before_exit':True}
(W/'clinical_exit_audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a),flush=True)
