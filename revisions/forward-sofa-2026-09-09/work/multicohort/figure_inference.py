"""Conditional exploratory inference for the final figures, using actual held-out scores.

Resampling estimates uncertainty conditional on the fitted models and trajectory labels;
it does not estimate full training-procedure uncertainty. The normal statistic uses
the observed paired difference divided by its bootstrap standard error. Patient
resampling is outcome-stratified in MIMIC; eICU resamples whole hospitals.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2_contingency
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

W = Path('work/multicohort')
R = W / 'results'
B = 10000
SEED = 20260909
METRICS = ['auroc', 'auprc', 'brier']


class WeightedMetrics:
    """Exact weighted empirical AUROC/AP with grouped ties, avoiding repeated sorts."""
    def __init__(self, y, score):
        self.y = np.asarray(y, dtype=float)
        self.score = np.asarray(score, dtype=float)
        self.order = np.argsort(-self.score, kind='stable')
        self.starts = np.r_[0, 1 + np.flatnonzero(np.diff(self.score[self.order]))]
        self.ys = self.y[self.order]
        self.loss = (self.score - self.y) ** 2

    def __call__(self, weights):
        w = weights[self.order]
        pos = np.add.reduceat(w * self.ys, self.starts)
        neg = np.add.reduceat(w * (1-self.ys), self.starts)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        auc = np.sum(pos * (fp[-1] - fp + neg/2)) / (tp[-1]*fp[-1])
        precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
        ap = np.sum(pos * precision) / tp[-1]
        return np.array([auc, ap, np.dot(weights, self.loss) / weights.sum()])


def holm(values):
    p = np.asarray(values)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def run(cohort, p, outcome, pairs, groups=None, seed=SEED):
    y = p[outcome].to_numpy(dtype=int)
    assert np.isfinite(p[[outcome]+sorted(set(sum(([a,b] for a,b in pairs), [])))].to_numpy()).all()
    models = sorted(set(sum(([a,b] for a,b in pairs), [])))
    functions = {m: WeightedMetrics(y, p[m]) for m in models}
    ones = np.ones(len(y))
    actual = {m: functions[m](ones) for m in models}
    # Verify the optimized metric implementation against independent sklearn routines,
    # including repeated observations and ties through integer sample weights.
    qrng = np.random.default_rng(816)
    for m, fn in functions.items():
        for weights in [ones, qrng.integers(0, 5, len(y)).astype(float)]:
            reference = np.array([roc_auc_score(y, p[m], sample_weight=weights),
                average_precision_score(y, p[m], sample_weight=weights),
                brier_score_loss(y, p[m], sample_weight=weights)])
            np.testing.assert_allclose(fn(weights), reference, rtol=1e-12, atol=1e-12)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == k) for k in [0,1]]
    if groups is not None:
        units, inverse = np.unique(groups, return_inverse=True)
    samples = np.empty((B, len(pairs), 3))
    for rep in range(B):
        if groups is None:
            ids = np.r_[rng.choice(strata[0],len(strata[0]),replace=True),
                        rng.choice(strata[1],len(strata[1]),replace=True)]
            weights = np.bincount(ids, minlength=len(y)).astype(float)
        else:
            weights = np.bincount(rng.integers(len(units),size=len(units)),
                                 minlength=len(units))[inverse].astype(float)
        vals = {m: fn(weights) for m,fn in functions.items()}
        for j,(a,b) in enumerate(pairs):
            samples[rep,j] = vals[a] - vals[b]
        if (rep+1) % 2500 == 0:
            print(cohort, rep+1, 'paired resamples', flush=True)
    assert np.isfinite(samples).all()
    rows=[]
    for j,(a,b) in enumerate(pairs):
        difference = actual[a] - actual[b]
        se = samples[:,j].std(axis=0, ddof=1)
        lo,hi = np.quantile(samples[:,j], [.025,.975], axis=0)
        for k,metric in enumerate(METRICS):
            if cohort == 'trajectory' and metric != 'auroc':
                continue
            z = difference[k] / se[k]
            rows.append(dict(cohort=cohort, model_a=a, model_b=b, metric=metric,
                n=len(y), events=int(y.sum()), difference=difference[k],
                ci_low=lo[k], ci_high=hi[k], bootstrap_se=se[k], z=z,
                p_raw=2*norm.sf(abs(z)), bootstrap_repeats=B, seed=seed,
                resampling_unit='hospital' if groups is not None else 'outcome-stratified patient',
                resampling_units=len(units) if groups is not None else len(y),
                family='trajectory_3_auroc_contrasts' if cohort=='trajectory' else 'mortality_9_metric_contrasts'))
    return rows


def main():
    rows=[]
    trajectory = pd.read_pickle(W/'mimiciii/private/trajectory_oof_predictions.pkl')
    pairs=[('truncated_posterior_24h','sofa_without_posterior'),
           ('truncated_posterior_12h','sofa_without_posterior'),
           ('truncated_posterior_24h','no_sofa_inputs')]
    rows += run('trajectory',trajectory,'target',pairs)
    for j,name in enumerate(['mimiciii','mimiciv','eicu'],1):
        groups=None
        if name=='mimiciv':
            raw=np.load('work/reanalysis/results/local_predictions.npz')
            p=pd.DataFrame({'death':raw['y'],'history':raw['history_4bin'],'latest':raw['last_4bin']})
        else:
            p=pd.read_pickle(W/f'private/{name}_mortality_oof.pkl')
        if name=='eicu':
            c=pd.read_pickle(W/'private/eicu_cohort.pkl').set_index('patientunitstayid').loc[p.index]
            assert c.index.equals(p.index)
            assert np.array_equal(c.hospitaldischargestatus.eq('Expired').astype(int),p.death)
            assert c.uniquepid.is_unique
            groups=c.hospitalid.to_numpy()
            assert len(np.unique(groups))==195
        rows += run(name,p,'death',[('history','latest')],groups,SEED+j)
    d=pd.DataFrame(rows)
    for family,ix in d.groupby('family').groups.items():
        d.loc[ix,'p_holm']=holm(d.loc[ix,'p_raw'])
    d.to_csv(R/'figure_paired_inference.csv',index=False)
    mortality=pd.read_csv(R/'mimiciii_mortality_by_trajectory.csv')
    tests=[]
    for k in [3,5]:
        sub=mortality.loc[mortality.k.eq(k)]
        contingency=np.c_[sub.deaths,sub.n-sub.deaths]
        chi,pv,df,expected=chi2_contingency(contingency,correction=False)
        tests.append(dict(k=k,n=int(sub.n.sum()),deaths=int(sub.deaths.sum()),
            chi_square=chi,df=df,p_raw=pv,min_expected_cell=expected.min(),
            cramers_v=np.sqrt(chi/sub.n.sum())))
    t=pd.DataFrame(tests);t['p_holm']=holm(t.p_raw)
    t.to_csv(R/'trajectory_mortality_association_tests.csv',index=False)
    audit=dict(seed_base=SEED,bootstrap_repeats=B,paired_tests=len(d),association_tests=len(t),
        metric_validation='Weighted empirical AUROC/AP/Brier match sklearn with unit and random integer weights.',
        test='Two-sided normal approximation: z = observed paired difference / bootstrap SD; P = 2*normal survival(abs(z)).',
        adjustment='Holm within 3 trajectory AUROC contrasts, 9 mortality metric contrasts, and 2 class-count mortality association tests.',
        scope='Exploratory, added after model evaluation. Frozen out-of-fold predictions and labels; no full-pipeline training uncertainty. No cross-database independence assumption.',
        intervals='10,000-resample paired percentile intervals supersede the earlier 1,000-resample paired intervals for the displayed contrasts. Individual-model intervals remain from 1,000 resamples.',
        association='Pearson chi-square, global K-by-2 table, fixed full-window assigned classes; not prospective validation. No SOFA-curve separation or BIC P tests.',
        normal_bootstrap_method_reference='https://xrobin.github.io/pROC/files/pROC_1.4.9_S_manual.pdf pp.55-56; independently implemented in Python, with whole-hospital resampling for eICU.')
    (R/'figure_inference_audit.json').write_text(json.dumps(audit,indent=2))
    print(d[['cohort','metric','model_a','model_b','difference','ci_low','ci_high','p_raw','p_holm']].to_string(index=False),flush=True)
    print(t.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
