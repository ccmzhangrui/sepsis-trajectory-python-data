from pathlib import Path
import json,os,time
import duckdb

W=Path('work/mimiciii');P=W/'private';c=duckdb.connect(str(P/'analysis.duckdb'))
c.execute("SET threads=6; SET memory_limit='4GB'")
# Timed IV antibacterial administrations are directly available in MetaVision.
# This avoids assigning artificial intraday times to prescription start dates.
ABX=[225850,225851,225853,225855,225859,225860,225862,225863,225865,225866,225875,225876,225879,225881,225883,225884,225886,225888,225889,225890,225892,225893,225898,225899,225900,225902,225798,225840,225842,225843,225845,225847,227691]
def make(name,sql):
    print(json.dumps({'start':name}),flush=True)
    c.execute('CREATE OR REPLACE TABLE mimiciii_derived.'+name+' AS '+sql)
    print(json.dumps({'table':name,'rows':c.execute('select count(*) from mimiciii_derived.'+name).fetchone()[0]}),flush=True)
make('first_adult_mv', '''WITH ranked AS (
 SELECT i.*,d.admission_age,d.gender,d.ethnicity,d.hospital_expire_flag,a.deathtime,a.admission_type,
 row_number() OVER(PARTITION BY i.subject_id ORDER BY i.intime,i.icustay_id) AS subject_icu_order
 FROM mimiciii.icustays i JOIN mimiciii_derived.icustay_detail d USING(icustay_id)
 JOIN mimiciii.admissions a ON i.hadm_id=a.hadm_id)
 SELECT * FROM ranked WHERE subject_icu_order=1 AND admission_age>=18
 AND dbsource='metavision' AND intime IS NOT NULL AND outtime>intime''')
make('abx_actual',f'''SELECT DISTINCT i.icustay_id,i.hadm_id,i.itemid,i.starttime AS antibiotic_time
 FROM mimiciii.inputevents_mv i JOIN mimiciii_derived.first_adult_mv f USING(icustay_id)
 WHERE i.itemid IN ({','.join(map(str,ABX))}) AND i.ordercategoryname='08-Antibiotics (IV)'
 AND i.statusdescription<>'Rewritten' AND i.amount>0
 AND i.starttime>=f.intime AND i.starttime<=least(f.outtime,f.intime+INTERVAL '48h')''')
make('culture_timed', '''SELECT DISTINCT hadm_id,charttime AS culture_time,spec_type_desc
 FROM mimiciii.microbiologyevents
 WHERE charttime IS NOT NULL AND (
 spec_type_desc LIKE '%BLOOD CULTURE%' OR spec_type_desc IN ('URINE','SPUTUM','BRONCHOALVEOLAR LAVAGE','PERITONEAL FLUID','CSF;SPINAL FLUID','ABSCESS','FLUID,OTHER','PLEURAL FLUID','BILE','JOINT FLUID','BRONCHIAL WASHINGS','TRACHEAL ASPIRATE'))''')
make('infection_pairs', '''SELECT DISTINCT a.icustay_id,a.hadm_id,a.antibiotic_time,b.culture_time,
 least(a.antibiotic_time,b.culture_time) AS suspicion_time,
 greatest(a.antibiotic_time,b.culture_time) AS infection_recognition_time
 FROM mimiciii_derived.abx_actual a JOIN mimiciii_derived.culture_timed b
 ON a.hadm_id=b.hadm_id AND abs(epoch(a.antibiotic_time-b.culture_time))<=86400
 JOIN mimiciii_derived.first_adult_mv f ON a.icustay_id=f.icustay_id
 WHERE greatest(a.antibiotic_time,b.culture_time)<=least(f.outtime,f.intime+INTERVAL '48h')
 AND least(a.antibiotic_time,b.culture_time)>=f.intime-INTERVAL '24h' ''')
make('sepsis_qualification', '''WITH q AS (
 SELECT p.*,h.endtime AS sofa_time,h.sofa_total,h.observed_components,
 greatest(p.infection_recognition_time,h.endtime) AS recognition_time,
 row_number() OVER(PARTITION BY p.icustay_id ORDER BY greatest(p.infection_recognition_time,h.endtime),p.suspicion_time,h.endtime) AS rn
 FROM mimiciii_derived.infection_pairs p JOIN mimiciii_derived.sofa_hourly h USING(icustay_id)
 WHERE h.endtime BETWEEN p.suspicion_time-INTERVAL '48h' AND p.suspicion_time+INTERVAL '24h'
 AND h.sofa_total>=2 AND h.observed_components>=4)
 SELECT * EXCLUDE(rn) FROM q WHERE rn=1''')
# This is an explicitly reconstructed, retrospective operational sepsis cohort.
# Total SOFA>=2 assumes baseline zero; it is NOT claimed to reproduce a measured
# preinfection Delta-SOFA>=2 or the original study's exact cohort.
make('analysis_cohort', '''SELECT f.*,s.suspicion_time,s.recognition_time,s.sofa_time,
 s.sofa_total AS qualifying_sofa,epoch(s.recognition_time-f.intime)/3600 AS recognition_hour
 FROM mimiciii_derived.first_adult_mv f JOIN mimiciii_derived.sepsis_qualification s USING(icustay_id)
 WHERE s.recognition_time<=least(f.outtime,f.intime+INTERVAL '48h')''')
flow={}
for name in ['icustays','first_adult_mv','abx_actual','infection_pairs','sepsis_qualification','analysis_cohort']:
    schema='mimiciii' if name=='icustays' else 'mimiciii_derived'
    flow[name]=c.execute(f'select count(distinct icustay_id) from {schema}.{name}').fetchone()[0]
landmarks=c.execute('''SELECT t,count(*) n FROM mimiciii_derived.analysis_cohort c CROSS JOIN (VALUES(6),(12),(24),(48)) v(t)
 WHERE recognition_hour<=t AND outtime>intime+t*INTERVAL '1h'
 GROUP BY t ORDER BY t''').fetchdf().to_dict('records')
cohort=c.execute('select * from mimiciii_derived.analysis_cohort ORDER BY subject_id').fetchdf()
cohort.to_pickle(P/'cohort.pkl');os.chmod(P/'cohort.pkl',0o600)
seq=c.execute('''SELECT s.* FROM mimiciii_derived.sofa_hourly s JOIN mimiciii_derived.analysis_cohort c USING(icustay_id)
 WHERE s.hour<=96 ORDER BY s.icustay_id,s.hour''').fetchdf()
seq.to_pickle(P/'sofa_hourly.pkl');os.chmod(P/'sofa_hourly.pkl',0o600)
abxdict=c.execute(f"select itemid,label from mimiciii.d_items where itemid in ({','.join(map(str,ABX))}) order by itemid").fetchdf().to_dict('records')
audit={'flow':flow,'landmarks':landmarks,'antibacterial_item_dictionary':abxdict,'cohort_definition':'First adult ICU stay in MetaVision; actual IV antibacterial administration plus timed body-fluid culture within 24h; total original SOFA >=2 with >=4 observed systems in [-48,+24]h around suspicion; both qualifying evidence sources known by landmark. Baseline zero assumption, not a measured acute Delta-SOFA reproduction.','warning_endpoint_author_confirmation':'The original operational event rule and linked onset records are unavailable; sepsis recognition is not substituted for original deterioration onset.','score_start':'ICU admission','label_window_hours':[6,60],'primary_landmark_hour':24,'primary_arterial_specimen':'ART only'}
(W/'cohort_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(audit,ensure_ascii=False),flush=True);c.close()
