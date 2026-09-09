"""Landmark (2/4/8/12/24/48 h) operational-deterioration prediction.

Uses the established model settings, nested CV and inner Youden thresholds.
Raw time filtering precedes aggregation and hourly histories are sorted.
The 24-hour primary results are reused exactly; landmarks <24 h reuse the same (24,60] h outcome; landmark 48 h uses
the at-risk subset with outcome window (48,60] h.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

BASE = Path.cwd()
sys.path.insert(0, str(BASE / 'work/multicohort/mimiciii'))
from gbtm import CensoredGBTM
sys.path.insert(0, str(BASE / 'work/multicohort'))
from figure_inference import WeightedMetrics

W = BASE / 'work/multicohort'
P = W / 'private'
R = W / 'results'
M = BASE / 'work/multicohort/mimiciii/private'
SEED = 20260913
LANDMARKS = [2, 4, 8, 12, 24, 48]
VITAL_ITEMIDS = "(211,220045,615,618,220210,224690,646,220277,676,678,223761,223762)"


def statistics(d, t):
    return {'last': d[-1] if len(d) else np.nan,
            'mean': d.mean() if len(d) else np.nan,
            'sd': d.std(ddof=1) if len(d) > 1 else np.nan,
            'min': d.min() if len(d) else np.nan,
            'max': d.max() if len(d) else np.nan,
            'slope': np.polyfit(t, d, 1)[0] if len(d) > 1 and np.ptp(t) > 0 else np.nan,
            'n_hours': len(d)}


def build_vital_hourly(index):
    """Filter raw chart AND store times at each cutoff BEFORE hourly aggregation."""
    c = duckdb.connect(str(BASE/'work/mimiciii/private/analysis.duckdb'), read_only=True)
    c.execute("SET threads=4; SET memory_limit='4GB'")
    eligible = pd.DataFrame({'icustay_id': index})
    c.register('eligible', eligible)
    out = c.execute(f"""
    WITH raw AS (
      SELECT z.icustay_id, i.intime, z.charttime, coalesce(z.storetime,z.charttime) AS available_time,
       ceil(epoch(z.charttime-i.intime)/3600)::integer AS hour,
       CASE WHEN itemid IN (211,220045) THEN 'hr' WHEN itemid IN(615,618,220210,224690) THEN 'rr'
        WHEN itemid IN(646,220277) THEN 'spo2' ELSE 'temp' END AS variable,
       CASE WHEN itemid IN(678,223761) THEN (valuenum-32)*5/9 ELSE valuenum END AS value
      FROM mimiciii.chartevents z JOIN mimiciii_derived.analysis_cohort i USING(icustay_id)
       JOIN eligible e USING(icustay_id)
      WHERE z.itemid IN {VITAL_ITEMIDS} AND (z.error IS NULL OR z.error<>1)
       AND z.charttime>=i.intime AND z.charttime<=i.intime+INTERVAL '48h'
       AND coalesce(z.storetime,z.charttime)<=i.intime+INTERVAL '48h'
    ), cutoffs AS (SELECT * FROM (VALUES (2),(4),(6),(8),(12),(24),(48)) AS t(cutoff))
    SELECT cutoff AS landmark_h,icustay_id,hour,variable,median(value) AS value
    FROM raw CROSS JOIN cutoffs
    WHERE charttime<=intime+cutoff*INTERVAL '1h' AND available_time<=intime+cutoff*INTERVAL '1h'
     AND ((variable='hr' AND value>0 AND value<300) OR (variable='rr' AND value>0 AND value<70)
      OR (variable='spo2' AND value>0 AND value<=100) OR (variable='temp' AND value>10 AND value<50))
    GROUP BY cutoff,icustay_id,hour,variable ORDER BY cutoff,icustay_id,hour,variable
    """).fetchdf()
    c.close()
    return out


def landmark_features(t, cohort, vitals, sofa, f24):
    """Feature matrix with information up to hour t. At t=24 returns the locked matrix."""
    if t == 24:
        return f24.loc[cohort.index]
    f = pd.DataFrame(index=cohort.index)
    f['age_topcoded90'] = f24['age_topcoded90']
    f['female'] = f24['female']
    vv = vitals.loc[vitals.landmark_h.eq(t)].sort_values(['icustay_id','hour'])
    for var in ['hr', 'rr', 'spo2', 'temp']:
        rows = {i: statistics(g.value.to_numpy(), g.hour.to_numpy())
                for i, g in vv.loc[vv.variable.eq(var)].groupby('icustay_id')}
        temp = pd.DataFrame.from_dict(rows, orient='index').reindex(f.index)
        temp.columns = [var + '_' + x for x in temp.columns]
        f = f.join(temp)
        f[var + '_n_hours'] = f[var + '_n_hours'].fillna(0)
    s = sofa.loc[sofa.hour.le(t) & sofa.observed_components.ge(4)].sort_values(['icustay_id','hour'])
    rows = {i: statistics(g.sofa_total.to_numpy(), g.hour.to_numpy()) for i, g in s.groupby('icustay_id')}
    sf = pd.DataFrame.from_dict(rows, orient='index').reindex(f.index)
    sf.columns = ['sofa_' + x for x in sf.columns]
    f = f.join(sf)
    components = ['respiration', 'coagulation', 'liver', 'cardiovascular', 'cns', 'renal', 'observed_components']
    last = sofa.loc[sofa.hour.eq(t)].set_index('icustay_id')[components].reindex(f.index)
    last.columns = ['sofa_' + x + '_24h' for x in last.columns]  # keep locked column names
    f = f.join(last)
    return f


def threshold(y, p):
    fpr, tpr, th = roc_curve(y, p)
    q = tpr - fpr
    q[~np.isfinite(th)] = -np.inf
    return float(th[np.argmax(q)])


def run_landmark(t, cohort, f, tmat, y, traj_grid):
    names = {'latest': ['age_topcoded90', 'female'] + [v + '_last' for v in ['hr', 'rr', 'temp', 'spo2']],
             'history': ['age_topcoded90', 'female'] + [v + '_' + z for v in ['hr', 'rr', 'temp', 'spo2'] for z in ['last', 'mean', 'sd', 'slope', 'n_hours']],
             'sofa_history': None, 'prefix_trajectory': None}
    sofa_cols = [x for x in f if x.startswith('sofa_')]
    names['sofa_history'] = names['history'] + sofa_cols
    names['prefix_trajectory'] = names['history'] + sofa_cols
    if traj_grid is None:
        names.pop('prefix_trajectory')
    predictions = {k: np.full(len(cohort), np.nan) for k in names}
    alarms = {k: np.zeros(len(cohort), bool) for k in names}
    yy = tmat.loc[cohort.index, traj_grid].to_numpy() if traj_grid else None
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fits = 0
    fit_audit=[]; cuts=[]; fold_ids=np.full(len(cohort),-1)
    for fold, (tr, te) in enumerate(outer.split(f, y), 1):
        print(f'  landmark {t}h outer fold {fold}', flush=True)

        def one(itr, ite, seed):
            nonlocal fits
            post_tr = post_te = None
            if traj_grid:
                g = CensoredGBTM(k=3, degree=2, starts=3, max_iter=500, seed=seed).fit(yy[itr], traj_grid)
                fits += 1
                fit_audit.append(g.to_dict())
                post_tr = g.predict_proba(yy[itr], traj_grid)
                post_te = g.predict_proba(yy[ite], traj_grid)
            out = {}
            for k, cols in names.items():
                xt = f.iloc[itr][cols].to_numpy(dtype=float, na_value=np.nan)
                xv = f.iloc[ite][cols].to_numpy(dtype=float, na_value=np.nan)
                if k == 'prefix_trajectory':
                    xt = np.c_[xt, post_tr]
                    xv = np.c_[xv, post_te]
                model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                      HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=15,
                                                                     min_samples_leaf=30, l2_regularization=5,
                                                                     early_stopping=False, random_state=seed))
                model.fit(xt, y[itr])
                out[k] = model.predict_proba(xv)[:, 1]
            return out

        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + fold)
        ip = {k: np.full(len(tr), np.nan) for k in names}
        for j, (itr, ite) in enumerate(inner.split(f.iloc[tr], y[tr]), 1):
            res = one(tr[itr], tr[ite], SEED + 100 * fold + j)
            for k in names:
                ip[k][ite] = res[k]
        res = one(tr, te, SEED + 100 * fold)
        for k in names:
            cut = threshold(y[tr], ip[k])
            predictions[k][te] = res[k]
            alarms[k][te] = res[k] >= cut
            cuts.append(dict(landmark_h=t,fold=fold,model=k,threshold=cut,n_train=len(tr),n_test=len(te)))
        fold_ids[te]=fold
    private=cohort[['event','event_hour']].copy();private['target']=y;private['fold']=fold_ids
    for k in names:private[k]=predictions[k];private[k+'_alarm']=alarms[k]
    private.to_pickle(P/f'landmark_corrected_oof_{t}h.pkl')
    pd.DataFrame(cuts).to_csv(R/f'landmark_corrected_thresholds_{t}h.csv',index=False)
    (R/f'landmark_corrected_audit_{t}h.json').write_text(json.dumps(dict(landmark_h=t,n=len(cohort),events=int(y.sum()),classifier_fits=20*len(names),trajectory_fits=fits,all_gbtm_converged=all(g['best_start']['em_converged'] for g in fit_audit),fitted_trajectories=fit_audit),indent=2))
    return names, predictions, alarms, fits


def summarize(t, cohort, y, names, predictions, alarms, scope):
    n = len(cohort)
    rows = []
    rng = np.random.default_rng(SEED + 1)
    strata = [np.flatnonzero(y == v) for v in [0, 1]]
    for k in names:
        fn = WeightedMetrics(y, predictions[k])
        samples = []
        for rep in range(1000):
            ids = np.r_[rng.choice(strata[0], len(strata[0]), True), rng.choice(strata[1], len(strata[1]), True)]
            w = np.bincount(ids, minlength=n).astype(float)
            samples.append(fn(w))
        actual = fn(np.ones(n))
        ci = np.quantile(samples, [.025, .975], axis=0)
        tn, fp, fnn, tp = confusion_matrix(y, alarms[k]).ravel()
        rows.append(dict(landmark_h=t, model=k, n=n, events=int(y.sum()),
                         auroc=actual[0], auroc_lo=ci[0, 0], auroc_hi=ci[1, 0],
                         auprc=actual[1], auprc_lo=ci[0, 1], auprc_hi=ci[1, 1],
                         brier=actual[2], brier_lo=ci[0, 2], brier_hi=ci[1, 2],
                         tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fnn), npv=tn/(tn+fnn), sensitivity=tp / (tp + fnn), specificity=tn / (tn + fp),
                         ppv=tp / (tp + fp) if tp + fp else np.nan, fp_per100=100 * fp / n, scope=scope))
    return rows


def main():
    cohort = pd.read_pickle(P / 'deterioration_cohort.pkl')
    f24 = pd.read_pickle(M / 'features_24h.pkl')
    tmat = pd.read_pickle(M / 'trajectory_matrix.pkl')
    sofa = pd.read_pickle(M / 'sofa_hourly.pkl')
    print('building hourly vitals 1-48h ...', flush=True)
    vitals = build_vital_hourly(cohort.index)
    vitals.to_pickle(P / 'landmark_corrected_vital_hourly.pkl')

    all_rows = []
    # ---- landmarks 2/4/8/12/24: same cohort, same (24,60] outcome ----
    y = cohort.event.to_numpy(dtype=int)
    for t in [2, 4, 8, 12, 24]:
        if t == 24:
            base=pd.read_csv(R/'deterioration_metrics.csv').rename(columns={'false_positives_per100':'fp_per100'})
            base['landmark_h']=24;base['scope']='Primary analysis reused without refitting or resampling; point estimates and intervals identical.'
            all_rows += base.to_dict('records')
            continue
        traj_grid = None
        f = landmark_features(t, cohort, vitals, sofa, f24)
        names, pred, alm, nf = run_landmark(t, cohort, f, tmat, y, traj_grid)
        scope = ('Corrected cutoff-first sorted histories in the selected hour-24 survivor cohort; predictors through hour %d; outcome unchanged (24,60] h.'
                 % t) + ('' if traj_grid else ' No trajectory-posterior model was evaluated before hour 24.')
        all_rows += summarize(t, cohort, y, names, pred, alm, scope)

    # ---- landmark 48: at-risk subset, outcome (48,60] ----
    risk = cohort.loc[(cohort.exit_hour > 48) & ~((cohort.event == 1) & (cohort.event_hour <= 48))].copy()
    y48 = ((risk.event == 1) & (risk.event_hour > 48)).to_numpy(dtype=int)
    print(f'landmark 48h at-risk n={len(risk)} events={y48.sum()}', flush=True)
    f48 = landmark_features(48, risk, vitals, sofa, f24.reindex(risk.index))
    names, pred, alm, nf = run_landmark(48, risk, f48, tmat, y48, [6, 12, 18, 24, 30, 36, 42, 48])
    all_rows += summarize(48, risk, y48, names, pred, alm,
                          'At-risk subset (event-free and under observation at hour 48); outcome (48,60] h; refolded with same seed.')

    out = pd.DataFrame(all_rows)
    out.to_csv(R / 'landmark_corrected_metrics.csv', index=False)
    print(out.to_string(index=False), flush=True)

    # ---- data availability at 24/48/60 and SOFA grid ----
    av = []
    for h in [2, 4, 6, 8, 12, 18, 24, 36, 48, 60]:
        observed = int((cohort.exit_hour > h).sum())
        evalu = int(sofa.loc[sofa.hour.eq(h) & sofa.observed_components.ge(4) & sofa.icustay_id.isin(cohort.index[cohort.exit_hour.gt(h)])].shape[0])
        av.append(dict(hour=h, under_observation_n=observed, evaluable_sofa_n=evalu))
    pd.DataFrame(av).to_csv(R / 'landmark_corrected_availability.csv', index=False)
    print(pd.DataFrame(av).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
