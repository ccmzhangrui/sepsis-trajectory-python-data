#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
# Use non-interactive backend to avoid display/survival-plot crashes
import matplotlib
matplotlib.use("Agg")
"""
End-to-end pipeline (synthetic reproducibility package):
- Ensure synthetic data exists
- Fit cubic SOFA trajectories (0/6/12/24/48h)
- Cluster coefficients with GMM; select K by BIC
- Export posterior membership probabilities + hard label
- Build engineered features
- Train binary model (demo: 28-day mortality)
- Survival: KM/logrank/Cox by trajectory group
- Save figures/results/models with stable feature ordering

Run:
    python sepsis_trajectory.py
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from src.io_utils import ensure_dirs, save_json
from src.synthetic_data import ensure_synthetic_csv
from src.trajectory import (
    TrajectoryConfig,
    fit_trajectory_gmm_bic,
    plot_sofa_trajectories,
    plot_pca_trajectories,
)
from src.features import FeatureConfig, build_feature_table
from src.modeling import ModelingConfig, train_binary_model, plot_roc_pr
from src.survival import SurvivalConfig, run_survival_analyses
from src.plots import plot_multi_param_trends

RANDOM_SEED = 42


def _load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "patient_id",
        "sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h",
        "lactate_0h", "lactate_6h", "lactate_12h", "lactate_24h", "lactate_48h",
        "age", "sex",
        "mortality_28d",
        "time_to_event_days",
        "event_observed",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    df["patient_id"] = df["patient_id"].astype(str)
    df["mortality_28d"] = df["mortality_28d"].astype(int)
    df["event_observed"] = df["event_observed"].astype(int)
    return df


def main() -> None:
    np.random.seed(RANDOM_SEED)
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    models_dir = root / "models"
    results_dir = root / "results"
    ensure_dirs([data_dir, models_dir, results_dir])

    # Ensure synthetic CSV exists
    csv_path = data_dir / "synthetic_sample_data.csv"
    ensure_synthetic_csv(csv_path, n=200, seed=RANDOM_SEED)
    df = _load_data(csv_path)

    # -------- Trajectory modeling (GMM + BIC)
    traj_cfg = TrajectoryConfig(
        timepoints_hours=(0, 6, 12, 24, 48),
        sofa_cols=("sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h"),
        degree=3,
        k_min=2,
        k_max=6,
        covariance_type="full",
        random_state=RANDOM_SEED,
    )
    traj_out = fit_trajectory_gmm_bic(df=df, cfg=traj_cfg, results_dir=results_dir)
    df_traj = df.merge(traj_out.posterior_df, on="patient_id", how="left")
    plot_sofa_trajectories(
        df=df_traj,
        cfg=traj_cfg,
        traj_label_col=traj_out.label_col,
        results_path=results_dir / "figure2_sofa_trajectories.png",
    )
    plot_pca_trajectories(
        coef_df=traj_out.coef_df,
        label_series=df_traj.set_index("patient_id")[traj_out.label_col],
        results_path=results_dir / "figure2_pca.png",
    )

    # -------- Feature engineering
    feat_cfg = FeatureConfig(
        timepoints_hours=(0, 6, 12, 24, 48),
        sofa_cols=("sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h"),
        lactate_cols=("lactate_0h", "lactate_6h", "lactate_12h", "lactate_24h", "lactate_48h"),
        include_entropy=True,
        include_spectral=True,
        random_state=RANDOM_SEED,
    )
    features_df, feature_names = build_feature_table(
        df=df_traj,
        cfg=feat_cfg,
        traj_p_cols=traj_out.posterior_cols,
        traj_label_col=traj_out.label_col,
    )
    save_json(models_dir / "feature_names.json", {"feature_names": feature_names})
    plot_multi_param_trends(
        df=df_traj,
        traj_label_col=traj_out.label_col,
        results_path=results_dir / "figure2_multi_param_trends.png",
    )

    # -------- Prediction modeling
    model_cfg = ModelingConfig(
        target_col="mortality_28d",
        id_col="patient_id",
        test_size=0.25,
        random_state=RANDOM_SEED,
        model_name="pipeline",
    )
    model_out = train_binary_model(
        features_df=features_df,
        feature_names=feature_names,
        cfg=model_cfg,
        models_dir=models_dir,
        results_dir=results_dir,
    )
    plot_roc_pr(
        model_out=model_out,
        results_dir=results_dir,
        roc_path=results_dir / "figure3_roc_multi_cohort.png",
        pr_path=results_dir / "figure3_pr_multi_cohort.png",
    )

    # -------- Survival analysis
    surv_cfg = SurvivalConfig(
        duration_col="time_to_event_days",
        event_col="event_observed",
        group_col=traj_out.label_col,
        id_col="patient_id",
        adjust_cols=("age", "sex"),
    )
    surv_out = run_survival_analyses(df=df_traj, cfg=surv_cfg, results_dir=results_dir)

    # -------- Save run summary
    summary = {
        "random_seed": RANDOM_SEED,
        "n_patients": int(df.shape[0]),
        "trajectory": {
            **asdict(traj_cfg),
            "selected_k": int(traj_out.selected_k),
            "label_col": traj_out.label_col,
            "posterior_cols": traj_out.posterior_cols,
        },
        "features": {**asdict(feat_cfg), "n_features": int(len(feature_names))},
        "modeling": {
            **asdict(model_cfg),
            "metrics": model_out.metrics,
            "model_path": str(model_out.model_path),
            "split_indices_path": str(model_out.split_indices_path),
        },
        "survival": {
            **asdict(surv_cfg),
            "logrank_p_value": surv_out.logrank_p_value,
            "cox_summary_path": str(surv_out.cox_summary_path),
            "km_plot_path": str(surv_out.km_plot_path),
        },
        "artifacts": {
            "bic_plot": str(results_dir / "bic_selection.png"),
            "sofa_traj_plot": str(results_dir / "figure2_sofa_trajectories.png"),
            "pca_plot": str(results_dir / "figure2_pca.png"),
            "trend_plot": str(results_dir / "figure2_multi_param_trends.png"),
            "roc_plot": str(results_dir / "figure3_roc_multi_cohort.png"),
            "pr_plot": str(results_dir / "figure3_pr_multi_cohort.png"),
        },
    }
    save_json(results_dir / "run_summary.json", summary)
    print("Done.")
    print(f"- Models:  {models_dir}")
    print(f"- Results: {results_dir}")


if __name__ == "__main__":
    main()
