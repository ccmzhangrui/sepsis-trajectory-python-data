from pathlib import Path
import csv,json,time,os,hashlib,re
import duckdb

ROOT=Path(os.environ['MIMICIII_DIR'])
WORK=Path('work/mimiciii');PRIVATE=WORK/'private';PRIVATE.mkdir(exist_ok=True);os.chmod(PRIVATE,0o700)
con=duckdb.connect(str(PRIVATE/'analysis.duckdb'))
con.execute("SET threads=6; SET memory_limit='4GB'; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{PRIVATE.resolve()}/spill'")
con.execute('CREATE SCHEMA IF NOT EXISTS mimiciii; CREATE SCHEMA IF NOT EXISTS mimiciii_derived')
audit=[]
def run(name,sql):
    start=time.time();print(json.dumps({'start':name}),flush=True);con.execute(sql)
    n=con.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
    record={'table':name,'rows':n,'seconds':round(time.time()-start,2)};audit.append(record)
    (WORK/'extraction_progress.json').write_text(json.dumps(audit,indent=2));print(json.dumps(record),flush=True)
def exists(name):return bool(con.execute("select count(*) from information_schema.tables where table_schema=? and table_name=?",name.split('.')).fetchone()[0])
def reader(name,alltext=False):
    return f"read_csv('{ROOT}/{name}.csv',header=true,sample_size=100000,nullstr=['','\\N'],all_varchar={'true' if alltext else 'false'},strict_mode=true)"
small=['patients','admissions','icustays','icustay_detail','sofa','angus','explicit','d_items','d_labitems','code_status','ventilation_durations','urine_output','norepinephrine_dose','epinephrine_dose','dopamine_dose','dobutamine_dose']
for name in small:
    schema='mimiciii_derived' if name in ['icustay_detail','sofa','angus','explicit','code_status','ventilation_durations','urine_output','norepinephrine_dose','epinephrine_dose','dopamine_dose','dobutamine_dose'] else 'mimiciii'
    full=f'{schema}.{name}'
    if not exists(full):run(full,f'CREATE TABLE {full} AS SELECT * FROM {reader(name)}')

# A single full CHARTEVENTS file is used. Named partitions are not concatenated.
items=[211,220045,51,52,443,456,6702,220052,220181,225312,198,184,454,723,220739,223900,223901,190,3420,3422,223835,615,618,220210,224690,646,220277,676,678,223761,223762]
full='mimiciii.chartevents'
if not exists(full):
    run(full,f'''CREATE TABLE {full} AS SELECT
       cast(row_id as bigint) row_id,cast(subject_id as bigint) subject_id,
       cast(hadm_id as bigint) hadm_id,cast(icustay_id as bigint) icustay_id,
       cast(itemid as integer) itemid,cast(charttime as timestamp) charttime,
       cast(storetime as timestamp) storetime,value,try_cast(valuenum as double) valuenum,
       valueuom,try_cast("error" as integer) AS "error"
       FROM {reader('chartevents',True)}
       WHERE cast(itemid as integer) IN ({','.join(map(str,items))}) AND icustay_id IS NOT NULL''')
labs=list(range(50800,50829))+[50862,50868,50882,50885,50902,50912,50931,50971,50983,51006,51144,51221,51222,51237,51265,51274,51275,51300,51301]
full='mimiciii.labevents'
if not exists(full):
    run(full,f'''CREATE TABLE {full} AS SELECT cast(row_id as bigint) row_id,
       cast(subject_id as bigint) subject_id,cast(hadm_id as bigint) hadm_id,
       cast(itemid as integer) itemid,cast(charttime as timestamp) charttime,
       value,try_cast(valuenum as double) valuenum,valueuom
       FROM {reader('labevents',True)} WHERE cast(itemid as integer) IN ({','.join(map(str,labs))})''')
for name in ['microbiologyevents','inputevents_mv','prescriptions']:
    full='mimiciii.'+name
    if not exists(full):run(full,f'CREATE TABLE {full} AS SELECT * FROM {reader(name)}')

# Audit chart source overlap using source row IDs, retaining only aggregate counts.
dupes=con.execute('SELECT count(*)-count(distinct row_id) FROM mimiciii.chartevents').fetchone()[0]
if dupes:raise ValueError(f'Selected CHARTEVENTS source contains {dupes} duplicate row IDs; investigate before scoring.')
part=con.execute(f"SELECT count(*) FROM (SELECT cast(row_id as bigint) row_id FROM {reader('chartevents_1',True)} LIMIT 1000) p JOIN mimiciii.chartevents c USING(row_id)").fetchone()[0]
summary={'duckdb_version':duckdb.__version__,'selected_chart_rows':con.execute('select count(*) from mimiciii.chartevents').fetchone()[0],'selected_chart_duplicate_row_ids':dupes,'partition_1_first_1000_overlap_with_selected_full_file':part,'partition_policy':'Read chartevents.csv once; do not concatenate the separately supplied numbered partitions.','tables':audit}
(WORK/'extraction_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
con.close();os.chmod(PRIVATE/'analysis.duckdb',0o600)
