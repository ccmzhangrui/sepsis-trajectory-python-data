from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd

# Match trajectory group display names and colors (same as Main_Figure1)
TRAJ_GROUP_NAMES = {0: "Rapid Recovery", 1: "Slow Recovery", 2: "Clinical Deterioration"}
TRAJ_COLORS = ["#4daf4a", "#ff7f00", "#e41a1c"]


def _traj_label(g) -> str:
    return TRAJ_GROUP_NAMES.get(int(g), f"Group {g}")


@dataclass(frozen=True)
class SurvivalConfig:
    duration_col: str = "time_to_event_days"
    event_col: str = "event_observed"
    group_col: str = "traj_label"
    id_col: str = "patient_id"
    adjust_cols: Tuple[str, ...] = ("age", "sex")


@dataclass(frozen=True)
class SurvivalOutput:
    logrank_p_value: float
    cox_summary_path: Path
    km_plot_path: Path


def run_survival_analyses(df: pd.DataFrame, cfg: SurvivalConfig, results_dir: Path) -> SurvivalOutput:
    results_dir.mkdir(parents=True, exist_ok=True)
    required = [cfg.duration_col, cfg.event_col, cfg.group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Survival df missing required columns: {missing}")
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    import matplotlib.pyplot as plt
    d = df.copy()
    d[cfg.event_col] = d[cfg.event_col].astype(int)
    d[cfg.duration_col] = pd.to_numeric(d[cfg.duration_col], errors="coerce")
    d = d.dropna(subset=[cfg.duration_col, cfg.event_col, cfg.group_col])
    lr = multivariate_logrank_test(
        event_durations=d[cfg.duration_col].to_numpy(),
        groups=d[cfg.group_col].astype(str).to_numpy(),
        event_observed=d[cfg.event_col].to_numpy(),
    )
    logrank_p = float(lr.p_value)
    km = KaplanMeierFitter()
    km_plot_path = results_dir / "figure4_km.png"
    plt.figure(figsize=(7.2, 5.2), dpi=160)
    for g in sorted(d[cfg.group_col].unique()):
        sub = d[d[cfg.group_col] == g]
        gi = int(g)
        c = TRAJ_COLORS[gi] if gi < len(TRAJ_COLORS) else None
        km.fit(sub[cfg.duration_col], event_observed=sub[cfg.event_col], label=f"{_traj_label(g)} (n={len(sub)})")
        km.plot_survival_function(ci_show=True, linewidth=2, color=c)
    plt.title(f"Kaplan–Meier by trajectory group (logrank p={logrank_p:.3g})")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.25)
    plt.legend(title="Trajectory", fontsize=10)
    plt.tight_layout()
    plt.savefig(km_plot_path)
    plt.close()
    cols = [cfg.duration_col, cfg.event_col, cfg.group_col, *cfg.adjust_cols]
    dd = d[cols].copy()
    for c in cfg.adjust_cols:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd = dd.dropna()
    dd = pd.get_dummies(dd, columns=[cfg.group_col], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(dd, duration_col=cfg.duration_col, event_col=cfg.event_col)
    cox_summary_path = results_dir / "cox_28d_summary.csv"
    cph.summary.reset_index().rename(columns={"index": "term"}).to_csv(cox_summary_path, index=False)
    return SurvivalOutput(logrank_p_value=logrank_p, cox_summary_path=cox_summary_path, km_plot_path=km_plot_path)
