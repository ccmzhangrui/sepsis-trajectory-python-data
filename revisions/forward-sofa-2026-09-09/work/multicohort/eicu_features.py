from pathlib import Path
import duckdb,json,os
import pandas as pd,numpy as np
W=Path('work/multicohort');P=W/'private';c=duckdb.connect(str(P/'eicu.duckdb'))
c.execute('''CREATE OR REPLACE TABLE four_vitals AS SELECT patientunitstayid,nursingchartoffset,nursingchartentryoffset,
 CASE WHEN nursingchartcelltypevallabel='Heart Rate' THEN 'hr' WHEN nursingchartcelltypevallabel='Respiratory Rate' THEN 'rr'
 WHEN nursingchartcelltypevallabel='O2 Saturation' THEN 'spo2' WHEN nursingchartcelltypevallabel='Temperature' THEN 'temp' END AS variable,
 CASE WHEN nursingchartcelltypevalname='Temperature (F)' THEN (try_cast(nursingchartvalue AS double)-32)*5/9 ELSE try_cast(nursingchartvalue AS double) END AS value
 FROM nurse24 WHERE (nursingchartcelltypevallabel IN ('Heart Rate','Respiratory Rate','O2 Saturation') AND nursingchartcelltypevalname=nursingchartcelltypevallabel)
 OR(nursingchartcelltypevallabel='Temperature' AND nursingchartcelltypevalname IN('Temperature (C)','Temperature (F)'))''')
c.execute('''CREATE OR REPLACE TABLE four_vital_hours AS SELECT patientunitstayid,ceil(nursingchartoffset/60.0)::integer AS hour,variable,median(value) AS value
 FROM four_vitals WHERE coalesce(nursingchartentryoffset,nursingchartoffset)<=1440
 AND((variable='hr' AND value>=20 AND value<=300) OR(variable='rr' AND value>=1 AND value<=80)
 OR(variable='spo2' AND value>=40 AND value<=100) OR(variable='temp' AND value>=25 AND value<=45))
 GROUP BY patientunitstayid,hour,variable''')
v=c.execute('select * from four_vital_hours order by patientunitstayid,hour').fetchdf();co=pd.read_pickle(P/'eicu_cohort.pkl').set_index('patientunitstayid')
f=pd.DataFrame(index=co.index);f['age_topcoded90']=np.minimum(co.age_numeric,90);f['female']=co.gender.map({'Female':1.,'Male':0.})
for var in ['hr','rr','spo2','temp']:
 rows={}
 for i,g in v.loc[v.variable.eq(var)].groupby('patientunitstayid'):
  d=g.value.to_numpy();t=g.hour.to_numpy();rows[i]={'last':d[-1],'mean':d.mean(),'sd':d.std(ddof=1) if len(d)>1 else np.nan,'slope':np.polyfit(t,d,1)[0] if len(d)>1 and np.ptp(t)>0 else np.nan,'n_hours':len(d)}
 z=pd.DataFrame.from_dict(rows,orient='index').reindex(f.index);z.columns=[var+'_'+x for x in z.columns];f=f.join(z);f[var+'_n_hours']=f[var+'_n_hours'].fillna(0)
f.to_pickle(P/'eicu_features.pkl');os.chmod(P/'eicu_features.pkl',0o600)
a=c.execute('''SELECT count(*) four_vital_records,sum(cast(nursingchartentryoffset>1440 AS integer)) late_entry_records,
 sum(cast(nursingchartentryoffset IS NULL AS integer)) missing_entry_offset FROM four_vitals''').fetchdf().to_dict('records')[0]
a.update({'n':len(f),'features':list(f),'completely_missing_vital_histories':int(f[[v+'_n_hours' for v in ['hr','rr','spo2','temp']]].sum(axis=1).eq(0).sum()),'gender_missing':int(f.female.isna().sum()),'window_minutes':[0,1440],'availability':'both measurement and entry offsets <=1440; missing entry offset falls back to measurement offset','temperature_rule':'convert Fahrenheit to Celsius, then median within hour','fixed_ranges':{'hr':[20,300],'rr':[1,80],'spo2':[40,100],'temp':[25,45]}})
(W/'eicu_feature_audit.json').write_text(json.dumps(a,indent=2));f.isna().mean().to_csv(W/'results/eicu_feature_missingness.csv');print(json.dumps(a),flush=True)
