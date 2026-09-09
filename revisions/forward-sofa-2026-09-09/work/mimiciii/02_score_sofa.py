from pathlib import Path
import json,time,os
import duckdb

W=Path('work/mimiciii');P=W/'private';con=duckdb.connect(str(P/'analysis.duckdb'))
con.execute("SET threads=6; SET memory_limit='5GB'; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{P.resolve()}/spill'")
audit=[]
def run(name,sql):
    t=time.time();print(json.dumps({'start':name}),flush=True);con.execute(sql)
    d={'table':name,'rows':con.execute(f'SELECT count(*) FROM {name}').fetchone()[0],'seconds':round(time.time()-t,2)}
    audit.append(d);(W/'scoring_progress.json').write_text(json.dumps(audit,indent=2));print(json.dumps(d),flush=True)
def exists(name):return bool(con.execute('select count(*) from information_schema.tables where table_schema=? and table_name=?',name.split('.')).fetchone()[0])
for name in ['pivoted_gcs','pivoted_bg','pivoted_lab','pivoted_bg_art']:
    full='mimiciii_derived.'+name
    if not exists(full):
        code=(W/'official'/f'mimic-iii__concepts_duckdb__pivot__{name}.sql').read_text()
        run(full,code)

# Direct 24-hour rolling aggregation avoids row multiplication and double rolling
# of a 24-hour urine sum in the public legacy pivoted_sofa query.
if not exists('mimiciii_derived.hour_grid'):
    run('mimiciii_derived.hour_grid','''CREATE TABLE mimiciii_derived.hour_grid AS
      SELECT i.subject_id,i.hadm_id,i.icustay_id,h AS hour,
        i.intime+h*INTERVAL '1h' AS endtime,i.intime,i.outtime
      FROM mimiciii.icustays i CROSS JOIN range(-23,169) h(h)
      WHERE i.intime IS NOT NULL AND i.outtime>i.intime
        AND i.intime+h*INTERVAL '1h' <= i.outtime''')

if not exists('mimiciii_derived.bp_hour'):
    run('mimiciii_derived.bp_hour','''CREATE TABLE mimiciii_derived.bp_hour AS
      SELECT c.icustay_id,cast(ceil(epoch(c.charttime-i.intime)/3600) AS integer) AS hour,min(c.valuenum) AS map
      FROM mimiciii.chartevents c JOIN mimiciii.icustays i USING(icustay_id)
      WHERE c.itemid IN (456,52,6702,443,220052,220181,225312) AND (c.error IS NULL OR c.error<>1)
       AND c.valuenum>0 AND c.valuenum<300
      GROUP BY c.icustay_id,hour''')
if not exists('mimiciii_derived.gcs_hour'):
    run('mimiciii_derived.gcs_hour','''CREATE TABLE mimiciii_derived.gcs_hour AS
      SELECT c.icustay_id,cast(ceil(epoch(c.charttime-i.intime)/3600) AS integer) AS hour,min(c.gcs) AS gcs
      FROM mimiciii_derived.pivoted_gcs c JOIN mimiciii.icustays i USING(icustay_id)
      WHERE c.gcs BETWEEN 3 AND 15 GROUP BY c.icustay_id,hour''')
if not exists('mimiciii_derived.lab_hour'):
    run('mimiciii_derived.lab_hour','''CREATE TABLE mimiciii_derived.lab_hour AS
      SELECT i.icustay_id,cast(ceil(epoch(c.charttime-i.intime)/3600) AS integer) AS hour,
        max(c.creatinine) AS creatinine,max(c.bilirubin) AS bilirubin,min(c.platelet) AS platelet,
        max(c.lactate) AS lactate
      FROM mimiciii_derived.pivoted_lab c JOIN mimiciii.icustays i USING(hadm_id)
      WHERE c.charttime>i.intime-INTERVAL '24h' AND c.charttime<=least(i.outtime,i.intime+INTERVAL '168h')
      GROUP BY i.icustay_id,hour''')
if not exists('mimiciii_derived.pf_hour'):
    run('mimiciii_derived.pf_hour','''CREATE TABLE mimiciii_derived.pf_hour AS
      WITH b AS (
        SELECT c.icustay_id,c.charttime,c.pao2fio2ratio,c.specimen,
          EXISTS(SELECT 1 FROM mimiciii_derived.ventilation_durations v
             WHERE v.icustay_id=c.icustay_id AND c.charttime>=v.starttime AND c.charttime<=v.endtime) AS vent
        FROM mimiciii_derived.pivoted_bg_art c
        WHERE c.pao2fio2ratio>0 AND c.pao2fio2ratio<=3000
      )
      SELECT b.icustay_id,cast(ceil(epoch(b.charttime-i.intime)/3600) AS integer) AS hour,
        min(CASE WHEN vent AND specimen='ART' THEN pao2fio2ratio END) AS pf_vent,
        min(CASE WHEN NOT vent AND specimen='ART' THEN pao2fio2ratio END) AS pf_novent,
        min(CASE WHEN vent THEN pao2fio2ratio END) AS pf_vent_inferred,
        min(CASE WHEN NOT vent THEN pao2fio2ratio END) AS pf_novent_inferred
      FROM b JOIN mimiciii.icustays i USING(icustay_id) GROUP BY b.icustay_id,hour''')
if not exists('mimiciii_derived.uo_hour'):
    run('mimiciii_derived.uo_hour','''CREATE TABLE mimiciii_derived.uo_hour AS
      SELECT u.icustay_id,cast(ceil(epoch(u.charttime-i.intime)/3600) AS integer) AS hour,
        sum(u.value) AS urineoutput,count(*) AS urine_records
      FROM mimiciii_derived.urine_output u JOIN mimiciii.icustays i USING(icustay_id)
      GROUP BY u.icustay_id,hour''')
for drug in ['norepinephrine','epinephrine','dopamine','dobutamine']:
    name='mimiciii_derived.'+drug+'_hour'
    if not exists(name):
        run(name,f'''CREATE TABLE {name} AS
          SELECT g.icustay_id,g.hour,max(d.vaso_rate) AS rate
          FROM mimiciii_derived.hour_grid g
          JOIN mimiciii_derived.{drug}_dose d ON g.icustay_id=d.icustay_id
            AND d.starttime<g.endtime AND d.endtime>g.endtime-INTERVAL '1h' AND d.vaso_rate>0
          GROUP BY g.icustay_id,g.hour''')
if not exists('mimiciii_derived.hour_raw'):
    run('mimiciii_derived.hour_raw','''CREATE TABLE mimiciii_derived.hour_raw AS
      SELECT g.*,b.map,c.gcs,l.creatinine,l.bilirubin,l.platelet,l.lactate,
        p.pf_vent,p.pf_novent,p.pf_vent_inferred,p.pf_novent_inferred,u.urineoutput,u.urine_records,
        n.rate AS norepinephrine,e.rate AS epinephrine,d.rate AS dopamine,o.rate AS dobutamine
      FROM mimiciii_derived.hour_grid g
      LEFT JOIN mimiciii_derived.bp_hour b USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.gcs_hour c USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.lab_hour l USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.pf_hour p USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.uo_hour u USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.norepinephrine_hour n USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.epinephrine_hour e USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.dopamine_hour d USING(icustay_id,hour)
      LEFT JOIN mimiciii_derived.dobutamine_hour o USING(icustay_id,hour)''')
if not exists('mimiciii_derived.rolling_variables'):
    expressions=[]
    for c in ['map','gcs','platelet','pf_vent','pf_novent','pf_vent_inferred','pf_novent_inferred']:
        expressions.append(f'min({c}) OVER win AS {c}')
    for c in ['creatinine','bilirubin','norepinephrine','epinephrine','dopamine','dobutamine','lactate']:
        expressions.append(f'max({c}) OVER win AS {c}')
    expressions += ["CASE WHEN hour>=24 AND sum(urineoutput) OVER win>=0 THEN sum(urineoutput) OVER win END AS urineoutput_24h",'count(urineoutput) OVER win AS urine_hours']
    run('mimiciii_derived.rolling_variables',f'''CREATE TABLE mimiciii_derived.rolling_variables AS
      SELECT subject_id,hadm_id,icustay_id,hour,endtime,intime,outtime,{','.join(expressions)}
      FROM mimiciii_derived.hour_raw WINDOW win AS(PARTITION BY icustay_id ORDER BY hour ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)''')
if not exists('mimiciii_derived.sofa_hourly'):
    # Match the original SOFA cut-offs; respiratory 1/2 also apply under ventilation.
    # Keep missing organ components explicit before constructing the conventional sum.
    sql='''CREATE TABLE mimiciii_derived.sofa_hourly AS WITH s AS (
      SELECT *,
       CASE WHEN pf_vent<100 THEN 4 WHEN pf_vent<200 THEN 3
         WHEN least(pf_vent,pf_novent)<300 THEN 2 WHEN least(pf_vent,pf_novent)<400 THEN 1
         WHEN coalesce(pf_vent,pf_novent) IS NOT NULL THEN 0 END AS respiration,
       CASE WHEN platelet<20 THEN 4 WHEN platelet<50 THEN 3 WHEN platelet<100 THEN 2
         WHEN platelet<150 THEN 1 WHEN platelet IS NOT NULL THEN 0 END AS coagulation,
       CASE WHEN bilirubin>=12 THEN 4 WHEN bilirubin>=6 THEN 3 WHEN bilirubin>=2 THEN 2
         WHEN bilirubin>=1.2 THEN 1 WHEN bilirubin IS NOT NULL THEN 0 END AS liver,
       CASE WHEN dopamine>15 OR epinephrine>0.1 OR norepinephrine>0.1 THEN 4
         WHEN dopamine>5 OR epinephrine>0 OR norepinephrine>0 THEN 3
         WHEN dopamine>0 OR dobutamine>0 THEN 2 WHEN map<70 THEN 1
         WHEN coalesce(map,norepinephrine,epinephrine,dopamine,dobutamine) IS NOT NULL THEN 0 END AS cardiovascular,
       CASE WHEN gcs<6 THEN 4 WHEN gcs<10 THEN 3 WHEN gcs<13 THEN 2 WHEN gcs<15 THEN 1
         WHEN gcs IS NOT NULL THEN 0 END AS cns,
       CASE WHEN creatinine>=5 OR urineoutput_24h<200 THEN 4
         WHEN creatinine>=3.5 OR urineoutput_24h<500 THEN 3
         WHEN creatinine>=2 THEN 2 WHEN creatinine>=1.2 THEN 1
         WHEN coalesce(creatinine,urineoutput_24h) IS NOT NULL THEN 0 END AS renal
      FROM mimiciii_derived.rolling_variables WHERE hour>=1)
      SELECT *,coalesce(respiration,0)+coalesce(coagulation,0)+coalesce(liver,0)+coalesce(cardiovascular,0)+coalesce(cns,0)+coalesce(renal,0) AS sofa_total,
       cast(respiration IS NOT NULL AS integer)+cast(coagulation IS NOT NULL AS integer)+cast(liver IS NOT NULL AS integer)+cast(cardiovascular IS NOT NULL AS integer)+cast(cns IS NOT NULL AS integer)+cast(renal IS NOT NULL AS integer) AS observed_components
      FROM s'''
    run('mimiciii_derived.sofa_hourly',sql)
counts=con.execute('''SELECT count(*) AS row_count,count(distinct icustay_id) stays,min(sofa_total) min_score,max(sofa_total) max_score,
 count(*)-count(distinct (icustay_id,hour)) duplicate_stay_hours,
 sum(cast(endtime>outtime AS integer)) post_exit_rows
 FROM mimiciii_derived.sofa_hourly''').fetchdf().to_dict('records')[0]
assert counts['duplicate_stay_hours']==0 and counts['post_exit_rows']==0
comparison=con.execute('''SELECT count(*) n,avg(h.sofa_total-s.sofa) mean_difference,
 avg(cast(h.sofa_total=s.sofa AS double)) exact_match,
 avg(abs(h.sofa_total-s.sofa)) mean_absolute_difference
 FROM mimiciii_derived.sofa_hourly h JOIN mimiciii_derived.sofa s USING(icustay_id) WHERE h.hour=24''').fetchdf().to_dict('records')[0]
(W/'sofa_scoring_audit.json').write_text(json.dumps({'counts':counts,'firstday_comparison':comparison,'steps':audit,'primary_arterial_specimen':'explicit ART','urine_rule':'24-hour volume only at ICU hour >=24; no extrapolation of short windows','score_window':'(endtime-24h,endtime]','missing_organs':'conventional zero contribution; observed_components retained','not_prospective_availability_validated':'LABEVENTS lacks storetime; charttime-aligned analyses cannot establish real-time release availability'},indent=2))
con.close();print(json.dumps({'counts':counts,'comparison':comparison}),flush=True)
