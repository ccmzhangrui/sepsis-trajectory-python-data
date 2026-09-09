# Forward SOFA trajectory prediction following the 9 September user correction

This protocol is written before fitting the forward models. The principal targets
remain future SOFA trajectories, not substituted mortality or organ-support events.
The existing source data support a reconstructed MIMIC-III analysis, not verified
membership in the original study cohorts.

Prediction origins are ICU T=12 and T=24 h. Source patients must have recognized
operational sepsis and remain alive/observed at T. Predictors use only recorded
information at or before T. Prefix trajectory grid: 6,12 h for T=12 (linear model),
and 6,12,18,24 h for T=24 (quadratic). The [0,T] interval is an information window;
there is no invented admission SOFA point at hour 0. At least two observed prefix
grid scores are required to construct a trajectory. Report those lacking a prefix.

Primary outcome grids are 18,24,...,60 h at T=12 and 30,36,...,60 h at T=24.
They exclude all prefix SOFA score observations. Scores remain conventional rolling
24-hour SOFA, so early outcome scores can include older source measurements. To
test this residual shared-window dependence, prespecified sensitivity outcome
grids start at T+24 h and end at60 h. Their rolling score windows exclude data
at/before T under the implemented (t-24,t] convention. This sensitivity has a later
target and stronger follow-up selection; it is not interchangeable with the main
target. No post-exit scores are imputed or extrapolated.

Evaluate future trajectories only when at least three future grid scores exist.
Keep prefix-eligible people with insufficient future observations in the prediction
pool, enumerate death/other exit/insufficient recording separately, and never label
them as non-deterioration or correct predictions. Train classifiers only where the
future label is assessable. Consequently performance conditions on assessability.

In every training partition fit a new 3-class censored-normal quadratic GBTM using
future-grid scores only. Apply that frozen future model to held-out future scores
to assign evaluation labels. Order classes by training-model expected SOFA at60 h.
Report all three class curves and one-versus-rest classification; the highest class
is called highest future SOFA, not automatically clinical deterioration. A rising
class is identified independently of prediction performance by fitted last-minus-
first future SOFA >0. Report rising-class results only if both outcome categories
exist in every relevant partition; if no rising class exists, report that absence
explicitly. No recentering or regrouping is introduced to manufacture rising classes.

Fit a separate 3-class prefix GBTM using only prefix data of training patients;
include all prefix-eligible training patients, irrespective of future assessability.
No future target/posterior enters the prefix fit. Five fixed patient KFold partitions
over the source cohort are reused across origins (seed20260919). Three inner patient
folds repeat future-label fitting, prefix fitting, imputation and classifier training;
inner pooled labels/predictions select finite Youden thresholds for each class.

Five classifier variants: vital history without SOFA (22 predictors); history plus
latest SOFA (23); that model plus historical SOFA slope (24); history plus latest
SOFA and prefix posterior probabilities (26); and prefix posteriors only (3).
The primary posterior increment compares the26-feature model to the23-feature model,
so this comparison does not include SOFA slope. The slope variant is an explicit
simple extrapolation comparator. All histories use cutoff-first raw chart/entry
filtering and sorted hours from the audited extractor. Laboratory release times
remain absent; SOFA availability is chart-time aligned, not fully verified real-time.

Fixed HistGradientBoosting settings:150iterations, learning rate0.05,15leaves,
minimum30perleaf,L2=5,noearlystopping. Median imputation with indicators is training-
only. GBTM:3starts,500iteration ceiling, best likelihood; document convergence and
class sizes. No outcome-based hyperparameter selection. Shared patient folds exclude
the same individual from training whenever that individual is evaluated.

Report class-specific AUROC/AUPRC/Brier with1000stratified bootstrap conditional CIs,
macro OVR AUROC and multiclass accuracy/log loss, confusion counts, PPV and coverage.
Compare posterior addition and slope addition to history+latest SOFA with10000paired
conditional bootstrap-normal tests; Holm correction across the12 primary contrasts
(2 origins x3classes x2additions). Sensitivity contrasts form a separate12-test family.
These intervals condition on learned future labels and fitted predictions, omitting
training uncertainty. Class persistence is not independent clinical warning.

Future trajectory membership has no unique onset timestamp. Do not invent a warning
interval from class membership, death or first future sampling time. Previously
computed support-event intervals remain explicitly secondary and cannot restore the
published serial17.6-hour SOFA deterioration warning-time estimate.
