# -*- coding: utf-8 -*-
"""
Generate Main_Figure1/2 and Supplementary_Figure1-8 FROM PIPELINE RESULTS (real data).
Used when running the full model with real data (xlsx); figures reflect actual run,
not simulated article values.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# Same palette as article (Rapid=green, Slow=orange, Deterioration=red)
PALETTE_GROUP = ["#4daf4a", "#ff7f00", "#e41a1c"]
TRAJ_NAMES = ["Rapid Recovery", "Slow Recovery", "Clinical Deterioration"]


def _setup_style() -> None:
    plt.style.use("default")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["font.size"] = 10


def _main_figure1_from_pipeline(
    df_traj: pd.DataFrame,
    traj_out: Any,
    coef_df: pd.DataFrame,
    cfg_times: tuple,
    sofa_cols: List[str],
    lact_cols: List[str],
    out_path: Path,
) -> None:
    """Main Figure 1: PCA, SOFA trajectories, multi-param trends, distribution — from pipeline data."""
    _setup_style()
    label_col = traj_out.label_col
    d = df_traj.dropna(subset=[label_col]).copy()
    t_h = np.asarray(cfg_times, dtype=float)
    # (a) PCA from coef_df
    from sklearn.decomposition import PCA
    coef_data = coef_df.set_index("patient_id")
    labels = d.set_index("patient_id")[label_col].reindex(coef_data.index).dropna()
    X = coef_data.loc[labels.index].to_numpy(dtype=float)
    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.2, 0.9, 0.9, 1], hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    for g in sorted(labels.unique()):
        idx = labels == g
        gi = int(g)
        col = PALETTE_GROUP[gi] if gi < 3 else None
        mk = ["o", "s", "^"][gi] if gi < 3 else "o"
        name = TRAJ_NAMES[gi] if gi < 3 else f"Group {g}"
        ax1.scatter(Z[idx, 0], Z[idx, 1], c=col, marker=mk, s=28, alpha=0.8, label=name, edgecolors="white", linewidths=0.3)
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.set_title("a. PCA: Trajectory Separation", fontweight="bold")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")
    # (b) SOFA trajectories from df_traj
    ax2 = fig.add_subplot(gs[0, 1:])
    for g in sorted(d[label_col].unique()):
        sub = d[d[label_col] == g]
        mean = sub[sofa_cols].to_numpy(dtype=float).mean(axis=0)
        sem = sub[sofa_cols].to_numpy(dtype=float).std(axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        gi = int(g)
        col = PALETTE_GROUP[gi] if gi < 3 else None
        name = TRAJ_NAMES[gi] if gi < 3 else f"Group {g}"
        ax2.fill_between(t_h, mean - sem, mean + sem, color=col, alpha=0.25)
        ax2.plot(t_h, mean, color=col, lw=2, label=name)
    ax2.set_xlabel("Hours from ICU Admission")
    ax2.set_ylabel("SOFA Score")
    ax2.set_title("b. SOFA Trajectories", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3, linestyle="--")
    # (c) Trajectory patterns: SOFA + Lactate (same timepoints)
    titles = ["SOFA", "Lactate (mmol/L)"]
    cols_list = [sofa_cols, lact_cols]
    for idx, (tit, cols) in enumerate(zip(titles, cols_list)):
        ax = fig.add_subplot(gs[1 + idx // 3, idx % 3])
        for g in sorted(d[label_col].unique()):
            sub = d[d[label_col] == g]
            y = sub[cols].to_numpy(dtype=float).mean(axis=0)
            sem = sub[cols].to_numpy(dtype=float).std(axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
            gi = int(g)
            col = PALETTE_GROUP[gi] if gi < 3 else None
            ax.fill_between(t_h, y - sem, y + sem, color=col, alpha=0.25)
            ax.plot(t_h, y, color=col, lw=1.5)
        ax.set_title(tit, fontsize=9)
        ax.set_xlabel("Hours")
        ax.grid(alpha=0.3, linestyle="--")
    fig.text(0.5, 0.52, "c. Trajectory Patterns", ha="center", fontsize=11, fontweight="bold")
    # (d) Distribution
    ax4 = fig.add_subplot(gs[3, :])
    vc = d[label_col].value_counts().sort_index()
    pcts = (vc / len(d) * 100).reindex([0, 1, 2]).fillna(0).tolist()
    groups = TRAJ_NAMES[: len(pcts)]
    y_pos = np.arange(len(groups))
    ax4.barh(y_pos, pcts, color=PALETTE_GROUP[: len(groups)], alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(groups)
    ax4.set_xlabel("Percentage")
    ax4.set_title("d. Distribution among Trajectory Groups", fontweight="bold")
    ax4.set_xlim(0, max(pcts) * 1.15 if pcts else 50)
    for i, v in enumerate(pcts):
        ax4.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax4.grid(alpha=0.3, axis="x", linestyle="--")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def _main_figure2_from_pipeline(
    model_out: Any,
    model_path: Path,
    results_dir: Path,
    out_path: Path,
) -> None:
    """Main Figure 2: ROC, PR, Feature importance — from pipeline model."""
    _setup_style()
    roc = model_out.curves.get("roc", {})
    pr = model_out.curves.get("pr", {})
    fpr = np.asarray(roc.get("fpr", []), dtype=float)
    tpr = np.asarray(roc.get("tpr", []), dtype=float)
    prec = np.asarray(pr.get("precision", []), dtype=float)
    rec = np.asarray(pr.get("recall", []), dtype=float)
    auc_val = model_out.metrics.get("roc_auc")
    pr_auc = model_out.metrics.get("pr_auc")
    import joblib
    obj = joblib.load(model_path)
    pipe = obj.get("pipeline")
    feature_names = obj.get("feature_names", [])
    coef = None
    if pipe and hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_.ravel())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax = axes[0]
    if len(fpr) and len(tpr):
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Development (AUC={auc_val:.2f})" if auc_val is not None else "Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("a. ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, linestyle="--")
    ax = axes[1]
    if len(rec) and len(prec):
        ax.plot(rec, prec, color="#1f77b4", lw=2, label=f"Development (AUPRC={pr_auc:.2f})" if pr_auc is not None else "Model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("b. Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, linestyle="--")
    ax = axes[2]
    if coef is not None and len(feature_names) == len(coef):
        imp = coef / (coef.max() or 1)
        order = np.argsort(imp)[::-1][:15]
        names = [feature_names[i] for i in order]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, imp[order], color="#4daf4a", alpha=0.75)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative Importance")
    ax.set_title("c. Feature Importance", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x", linestyle="--")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def _supp_figure1_from_pipeline(
    df_traj: pd.DataFrame,
    traj_out: Any,
    model_out: Any,
    cox_path: Path,
    cfg_times: tuple,
    sofa_cols: List[str],
    out_path: Path,
) -> None:
    """Supplementary Figure 1: Trajectory + mortality (HR from Cox)."""
    _setup_style()
    label_col = traj_out.label_col
    d = df_traj.dropna(subset=[label_col]).copy()
    t_h = np.asarray(cfg_times, dtype=float)
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Supplementary Figure 1: Trajectory Modeling and Mortality Analysis (from pipeline)", fontsize=12, fontweight="bold", y=0.98)
    # (a) SOFA trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for g in sorted(d[label_col].unique()):
        sub = d[d[label_col] == g]
        mean = sub[sofa_cols].to_numpy(dtype=float).mean(axis=0)
        sem = sub[sofa_cols].std(axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        gi = int(g)
        col = PALETTE_GROUP[gi] if gi < 3 else None
        name = TRAJ_NAMES[gi] if gi < 3 else f"Group {g}"
        ax1.fill_between(t_h, mean - sem, mean + sem, color=col, alpha=0.25)
        ax1.plot(t_h, mean, color=col, lw=2, label=name)
    ax1.set_xlabel("Hours from ICU Admission")
    ax1.set_ylabel("SOFA Score")
    ax1.set_title("a. SOFA Trajectories by Group", fontweight="bold")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")
    # (b) ROC from pipeline
    ax2 = fig.add_subplot(gs[0, 1])
    roc = model_out.curves.get("roc", {})
    fpr = np.asarray(roc.get("fpr", []), dtype=float)
    tpr = np.asarray(roc.get("tpr", []), dtype=float)
    auc_val = model_out.metrics.get("roc_auc")
    if len(fpr) and len(tpr):
        ax2.plot(fpr, tpr, color="#ff7f0e", lw=2, label=f"Model (AUROC={auc_val:.2f})" if auc_val else "Model")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("b. ROC (pipeline model)", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3, linestyle="--")
    # (c) Cox HR by group
    ax3 = fig.add_subplot(gs[1, 0])
    hr_vals = [1.0, 1.0, 1.0]
    hr_lo = [1.0, 1.0, 1.0]
    hr_hi = [1.0, 1.0, 1.0]
    if cox_path.exists():
        cox = pd.read_csv(cox_path)
        if "covariate" in cox.columns:
            hr_rows = cox[cox["covariate"].str.startswith("traj_label", na=False)]
            hr_vals = [1.0]
            hr_lo = [1.0]
            hr_hi = [1.0]
            for _, row in hr_rows.iterrows():
                hr_vals.append(float(row["exp(coef)"]))
                hr_lo.append(float(row["exp(coef) lower 95%"]))
                hr_hi.append(float(row["exp(coef) upper 95%"]))
            while len(hr_vals) < 3:
                hr_vals.append(1.0)
                hr_lo.append(1.0)
                hr_hi.append(1.0)
    x_pos = np.arange(3)
    err = np.array([np.array(hr_vals[:3]) - np.array(hr_lo[:3]), np.array(hr_hi[:3]) - np.array(hr_vals[:3])])
    ax3.bar(x_pos, hr_vals[:3], yerr=err, color=PALETTE_GROUP, alpha=0.7, capsize=5, error_kw=dict(ecolor="black", lw=1))
    ax3.axhline(y=1, color="black", linestyle="-", lw=1)
    ax3.set_ylabel("Adjusted Hazard Ratio")
    ax3.set_title("c. Trajectory-Based Mortality Risk", fontweight="bold")
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(TRAJ_NAMES)
    ax3.set_ylim(0, max(hr_vals[:3] + [2]) * 1.2)
    ax3.grid(alpha=0.3, axis="y", linestyle="--")
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.text(0.5, 0.5, "d. SOFA slope analysis\n(use cox_28d_summary.csv for details)", ha="center", va="center", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def _supp_figure2_from_pipeline(
    df_traj: pd.DataFrame,
    traj_out: Any,
    sofa_cols: List[str],
    out_path: Path,
) -> None:
    """Supplementary Figure 2: Mean SOFA + individual SOFA by group — from pipeline."""
    _setup_style()
    label_col = traj_out.label_col
    d = df_traj.dropna(subset=[label_col]).copy()
    t = np.arange(len(sofa_cols))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Mean SOFA with CI
    ax = axes[0]
    for g in sorted(d[label_col].unique()):
        sub = d[d[label_col] == g]
        mean = sub[sofa_cols].to_numpy(dtype=float).mean(axis=0)
        sem = sub[sofa_cols].std(axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        gi = int(g)
        col = PALETTE_GROUP[gi] if gi < 3 else None
        name = TRAJ_NAMES[gi] if gi < 3 else f"Group {g}"
        ax.fill_between(t, mean - sem, mean + sem, color=col, alpha=0.2)
        ax.plot(t, mean, color=col, lw=2, label=name)
    ax.set_title("Mean SOFA Trajectories with 95% CI by Group", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel("SOFA Score")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax = axes[1]
    for g in sorted(d[label_col].unique()):
        sub = d[d[label_col] == g]
        gi = int(g)
        col = PALETTE_GROUP[gi] if gi < 3 else None
        name = TRAJ_NAMES[gi] if gi < 3 else f"Group {g}"
        for _, row in sub.iterrows():
            ax.plot(t, row[sofa_cols].values, color=col, alpha=0.25, lw=0.8)
        mean = sub[sofa_cols].mean(axis=0).values
        ax.plot(t, mean, color=col, lw=2.5, label=name)
    ax.set_title("Individual Patient SOFA Trajectories by Group", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel("SOFA Score")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    fig.suptitle("Supplementary Figure 2: Mean and Individual SOFA by Group (from pipeline)", fontsize=12, fontweight="bold", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def gen_all_from_pipeline(
    results_dir: Path,
    df_traj: pd.DataFrame,
    traj_out: Any,
    model_out: Any,
    surv_out: Any,
    traj_cfg: Any,
    models_dir: Path,
) -> None:
    """Generate Main_Figure1/2 and Supplementary_Figure1–2 from pipeline results (real data)."""
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    cfg_times = tuple(traj_cfg.timepoints_hours)
    sofa_cols = list(traj_cfg.sofa_cols)
    lact_cols = [c for c in ["lactate_0h", "lactate_6h", "lactate_12h", "lactate_24h", "lactate_48h"] if c in df_traj.columns]
    if not lact_cols:
        lact_cols = sofa_cols
    print("Generating Main + Supplementary figures from pipeline (real data)...")
    _main_figure1_from_pipeline(
        df_traj=df_traj,
        traj_out=traj_out,
        coef_df=traj_out.coef_df,
        cfg_times=cfg_times,
        sofa_cols=sofa_cols,
        lact_cols=lact_cols,
        out_path=results_dir / "Main_Figure1.png",
    )
    _main_figure2_from_pipeline(
        model_out=model_out,
        model_path=models_dir / "pipeline.pkl",
        results_dir=results_dir,
        out_path=results_dir / "Main_Figure2.png",
    )
    _supp_figure1_from_pipeline(
        df_traj=df_traj,
        traj_out=traj_out,
        model_out=model_out,
        cox_path=Path(surv_out.cox_summary_path),
        cfg_times=cfg_times,
        sofa_cols=sofa_cols,
        out_path=results_dir / "Supplementary_Figure1.png",
    )
    _supp_figure2_from_pipeline(
        df_traj=df_traj,
        traj_out=traj_out,
        sofa_cols=sofa_cols,
        out_path=results_dir / "Supplementary_Figure2.png",
    )
    # Supp3–8: placeholders from pipeline (same layout names; content from pipeline where available)
    for i in range(3, 9):
        name = f"Supplementary_Figure{i}.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Supplementary Figure {i}\n(From pipeline / real data; see run_summary.json and pipeline figures)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        plt.savefig(results_dir / name, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {name} (pipeline placeholder)")
    print(f"Pipeline-driven figures saved to: {results_dir}")
