from pathlib import Path
import duckdb,json,time,os,hashlib
ROOT=Path(os.environ['EICU_DIR']);W=Path('work/multicohort');P=W/'private';c=duckdb.connect(str(P/'eicu.duckdb'))
c.execute("SET threads=4; SET memory_limit='3GB'")
c.execute(f"SET temp_directory='{P.resolve()}/eicu_spill'")
def csv(name):return f"read_csv('{ROOT}/{name}.csv',header=true,sample_size=-1,nullstr=['','NULL'],strict_mode=true)"
for name in ['patient','diagnosis']:
 c.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT * FROM {csv(name)}')
 print(name,c.execute(f'select count(*) from {name}').fetchone()[0],flush=True)
c.execute('''CREATE OR REPLACE TABLE dx_sepsis AS SELECT patientunitstayid,min(diagnosisoffset) sepsis_entry_offset
 FROM diagnosis WHERE diagnosisoffset BETWEEN 0 AND 1440
 AND(regexp_matches(lower(diagnosisstring),'(^|[| ])sepsis($|[| ])|septic shock')) GROUP BY patientunitstayid''')
# First ICU within a hospital stay; for repeat hospitalizations take one eligible stay
# using a deterministic outcome-independent digest (not a claim of first-ever admission).
c.execute('''CREATE OR REPLACE TABLE candidate AS WITH firstunit AS (
 SELECT *,row_number() OVER(PARTITION BY patienthealthsystemstayid ORDER BY hospitaladmitoffset DESC,patientunitstayid) first_rank,
 CASE WHEN trim(age)='> 89' THEN 90 ELSE try_cast(age AS double) END age_numeric
 FROM patient)
 SELECT p.*,d.sepsis_entry_offset FROM firstunit p JOIN dx_sepsis d USING(patientunitstayid)
 WHERE first_rank=1 AND age_numeric>=18 AND hospitaldischargestatus IN ('Alive','Expired')
 AND unitdischargeoffset>1440 AND hospitaldischargeoffset>1440
 AND uniquepid IS NOT NULL''')
c.execute('''CREATE OR REPLACE TABLE cohort AS SELECT * EXCLUDE(rn) FROM (
 SELECT *,row_number() OVER(PARTITION BY uniquepid ORDER BY md5(cast(patientunitstayid AS varchar)||'_20260908')) rn FROM candidate) WHERE rn=1''')
print('eligible',c.execute('select count(*),sum(cast(hospitaldischargestatus=\'Expired\' as integer)),count(distinct hospitalid) from cohort').fetchone(),flush=True)
# Read the original table once; retain chart and validation offsets separately.
if not c.execute("select count(*) from information_schema.tables where table_name='nurse24'").fetchone()[0]:
 c.execute(f'''CREATE TABLE nurse24 AS SELECT n.* FROM {csv('nurseCharting')} n
 JOIN cohort p USING(patientunitstayid) WHERE nursingchartoffset BETWEEN 0 AND 1440''')
dictionary=c.execute('''SELECT nursingchartcelltypecat AS category,nursingchartcelltypevallabel AS "label",nursingchartcelltypevalname AS name,count(*) n,
 sum(cast(try_cast(nursingchartvalue AS double) IS NOT NULL AS integer)) numeric_n
 FROM nurse24 GROUP BY 1,2,3 ORDER BY n DESC''').fetchdf()
dictionary.to_csv(W/'eicu_nursing_dictionary.csv',index=False)
c.execute('select * from cohort order by patientunitstayid').fetchdf().to_pickle(P/'eicu_cohort.pkl')
flow={k:c.execute(sql).fetchone()[0] for k,sql in {'source_stays':'select count(*) from patient','documented_sepsis_by24':'select count(*) from dx_sepsis','eligible_hospital_stays':'select count(*) from candidate','selected_unique_patients':'select count(*) from cohort','hospital_deaths':"select count(*) from cohort where hospitaldischargestatus='Expired'",'hospitals':'select count(distinct hospitalid) from cohort','nurse24_records':'select count(*) from nurse24'}.items()}
(W/'eicu_cohort_audit.json').write_text(json.dumps(flow,indent=2));print(json.dumps(flow),flush=True);c.close()
manifest=[]
for name in ['patient','diagnosis','nurseCharting']:
 p=ROOT/(name+'.csv');h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 manifest.append({'file':p.name,'size':p.stat().st_size,'sha256':h.hexdigest()})
(W/'eicu_source_manifest.json').write_text(json.dumps(manifest,indent=2));print('hashes complete',flush=True)
