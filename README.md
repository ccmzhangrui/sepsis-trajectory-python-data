# sepsis-trajectory-python-data

Runnable, end-to-end reproducibility package for the workflow described in:
"Machine Learning Predicts Sepsis Deterioration Trajectories".

This repository is designed to be **directly runnable** with a **synthetic dataset** (schema-compatible) so reviewers/readers can reproduce the *computational workflow* and regenerate figures/tables/model artifacts.

## What this repository provides

Pipeline components implemented in Python:

1) **Trajectory identification** (SOFA at 0/6/12/24/48h)
   - Fit **cubic polynomials** per patient (coefficients as trajectory representation)
   - Cluster using **Gaussian Mixture Models (GMM)** (probabilistic; soft assignment)
   - Select number of classes **K via BIC** (with a saved plot)
   - Export **posterior membership probabilities** for downstream prediction

2) **Feature engineering**
   - Trends/slopes/deltas (e.g., 0–24h, 0–48h)
   - Variability: mean/SD
   - Complexity: **Approximate Entropy + Sample Entropy** (demo-grade implementations)
   - Simple spectral features from FFT bandpower (demonstration)
   - Lactate clearance (0–24h, 0–48h)

3) **Binary prediction** (demo endpoint: 28-day mortality)
   - Standard preprocessing (imputation + scaling)
   - Logistic regression classifier (balanced)
   - Deterministic split; indices saved for reproducibility

4) **Survival analysis**
   - Kaplan–Meier curves by trajectory group
   - Log-rank test
   - Cox proportional hazards model summary saved to CSV

## Data availability and reproducibility note (important)

Real patient-level clinical data cannot be publicly released. This repository therefore includes (or can auto-generate) a **synthetic dataset** that matches the required structure and allows the pipeline to run end-to-end.

**Important:** Results obtained using the synthetic data are for computational reproducibility and **will not match the manuscript's performance estimates** (derived from restricted-access clinical cohorts).

## Quickstart

```bash
pip install -r requirements.txt
python sepsis_trajectory.py
```

## Outputs

Running `python sepsis_trajectory.py` will generate/overwrite:

- `models/`
  - `pipeline.pkl`
  - `feature_names.json`
  - `split_indices.json`
- `results/`
  - `bic_selection.png`
  - `figure2_sofa_trajectories.png`
  - `figure2_pca.png`
  - `figure2_multi_param_trends.png`
  - `figure3_roc_multi_cohort.png`
  - `figure3_pr_multi_cohort.png`
  - `figure4_km.png`
  - `cox_28d_summary.csv`
  - `model_metrics.json`
  - `run_summary.json`
  - `feature_importance.csv` (if explain step is run)

## Reproducibility

- A fixed random seed is used throughout.
- Train/test split indices are saved to `models/split_indices.json`.
- Feature names are saved to `models/feature_names.json` to ensure consistent ordering.

## Notes for reviewers

- This repository intentionally does **not** ship any private clinical data or pre-trained models derived from private cohorts.
- All model artifacts listed above are produced by running the pipeline locally on the included/auto-generated synthetic dataset.
