from pathlib import Path
import json,os
import duckdb,numpy as np,pandas as pd
W=Path('work/mimiciii');P=W/'private';c=duckdb.connect(str(P/'analysis.duckdb'))
c.execute("SET threads=6; SET memory_limit='4GB'")
y=pd.read_pickle(P/'trajectory_matrix.pkl');co=pd.read_pickle(P/'cohort.pkl').set_index('icustay_id').loc[y.index]
c.execute('''CREATE OR REPLACE TABLE mimiciii_derived.vital_hour_available AS
 WITH v AS (
 SELECT z.icustay_id,ceil(epoch(z.charttime-i.intime)/3600)::integer AS hour,
 CASE WHEN itemid IN (211,220045) THEN 'hr' WHEN itemid IN(615,618,220210,224690) THEN 'rr'
 WHEN itemid IN(646,220277) THEN 'spo2' WHEN itemid IN(676,678,223761,223762) THEN 'temp' END AS variable,
 CASE WHEN itemid IN(678,223761) THEN (valuenum-32)*5/9 ELSE valuenum END AS v
 FROM mimiciii.chartevents z JOIN mimiciii_derived.analysis_cohort i USING(icustay_id)
 WHERE z.itemid IN(211,220045,615,618,220210,224690,646,220277,676,678,223761,223762)
 AND(z.error IS NULL OR z.error<>1) AND z.charttime>=i.intime AND z.charttime<=i.intime+INTERVAL '24h'
 AND coalesce(z.storetime,z.charttime)<=i.intime+INTERVAL '24h')
 SELECT icustay_id,hour,variable,median(v) AS value FROM v
 WHERE (variable='hr' AND v>0 AND v<300) OR (variable='rr' AND v>0 AND v<70)
 OR(variable='spo2' AND v>0 AND v<=100) OR(variable='temp' AND v>10 AND v<50)
 GROUP BY icustay_id,hour,variable''')
v=c.execute('select * from mimiciii_derived.vital_hour_available order by icustay_id,hour').fetchdf()
f=pd.DataFrame(index=y.index);f['age_topcoded90']=np.minimum(co.admission_age,90);f['female']=co.gender.eq('F').astype(int)
def statistics(d,t):
    return {'last':d[-1] if len(d) else np.nan,'mean':d.mean() if len(d) else np.nan,'sd':d.std(ddof=1) if len(d)>1 else np.nan,'min':d.min() if len(d) else np.nan,'max':d.max() if len(d) else np.nan,'slope':np.polyfit(t,d,1)[0] if len(d)>1 and np.ptp(t)>0 else np.nan,'n_hours':len(d)}
for var in ['hr','rr','spo2','temp']:
    rows={i:statistics(g.value.to_numpy(),g.hour.to_numpy()) for i,g in v.loc[v.variable.eq(var)].groupby('icustay_id')}
    temp=pd.DataFrame.from_dict(rows,orient='index').reindex(f.index);temp.columns=[var+'_'+x for x in temp.columns];f=f.join(temp)
    f[var+'_n_hours']=f[var+'_n_hours'].fillna(0)
independent=list(f.columns)
s=pd.read_pickle(P/'sofa_hourly.pkl')
rows={i:statistics(g.sofa_total.to_numpy(),g.hour.to_numpy()) for i,g in s.loc[s.hour.le(24)&s.observed_components.ge(4)].groupby('icustay_id')}
sf=pd.DataFrame.from_dict(rows,orient='index').reindex(f.index);sf.columns=['sofa_'+x for x in sf.columns];f=f.join(sf)
components=['respiration','coagulation','liver','cardiovascular','cns','renal','observed_components']
last=s.loc[s.hour.eq(24)].set_index('icustay_id')[components].reindex(f.index);last.columns=['sofa_'+x+'_24h' for x in last.columns];f=f.join(last)
assert 'dead_time' not in f and 'deathtime' not in f and f.index.equals(y.index)
f.to_pickle(P/'features_24h.pkl');os.chmod(P/'features_24h.pkl',0o600)
late=c.execute('''SELECT count(*) n,sum(cast(z.storetime>i.intime+INTERVAL '24h' AS integer)) late_store_records,
 sum(cast(z.storetime IS NULL AS integer)) missing_storetime
 FROM mimiciii.chartevents z JOIN mimiciii_derived.analysis_cohort i USING(icustay_id)
 WHERE z.itemid IN(211,220045,615,618,220210,224690,646,220277,676,678,223761,223762)
 AND z.charttime BETWEEN i.intime AND i.intime+INTERVAL '24h' ''').fetchdf().to_dict('records')[0]
(W/'feature_audit.json').write_text(json.dumps({'n':len(f),'predictor_count':len(f.columns),'no_sofa_predictors':independent,'all_predictors':list(f.columns),'raw_vital_availability':late,'hourly_vitals':'median of valid values charted within first24h and stored by prediction time; missing storetime uses charttime','age':'topcoded at90','SOFA_predictor_caveat':'SOFA/labs aligned on charttime; laboratory release time absent; separate vital-only variant excludes all original SOFA inputs.'},indent=2));print(json.dumps({'n':len(f),'predictors':len(f.columns),'late_records':late}),flush=True)
c.close()
