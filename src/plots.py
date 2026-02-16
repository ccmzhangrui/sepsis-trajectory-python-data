from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Match trajectory group display names and colors (same as Main_Figure1)
TRAJ_GROUP_NAMES = {0: "Rapid Recovery", 1: "Slow Recovery", 2: "Clinical Deterioration"}
TRAJ_COLORS = ["#4daf4a", "#ff7f00", "#e41a1c"]


def _traj_label(g) -> str:
    return TRAJ_GROUP_NAMES.get(int(g), f"Group {g}")


def plot_roc_curve(roc_obj: Dict[str, Any], save_path: Path, auc: float = None) -> None:
    fpr = np.asarray(roc_obj.get("fpr", []), dtype=float)
    tpr = np.asarray(roc_obj.get("tpr", []), dtype=float)
    if auc is None and len(fpr) > 1 and len(tpr) > 1:
        auc = float(np.trapz(tpr, fpr))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 5.0), dpi=160)
    if len(fpr) and len(tpr):
        lbl = "Model"
        if auc is not None and not np.isnan(auc):
            lbl += f" (AUC = {auc:.3f})"
        plt.plot(fpr, tpr, linewidth=2, label=lbl)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.grid(alpha=0.25)
    if auc is not None and not np.isnan(auc):
        plt.text(0.05, 0.95, f"AUC = {auc:.3f}", fontsize=12, fontweight="bold",
                 transform=plt.gca().transAxes, va="top", ha="left",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    if len(fpr) and len(tpr):
        plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(pr_obj: Dict[str, Any], save_path: Path) -> None:
    precision = np.asarray(pr_obj.get("precision", []), dtype=float)
    recall = np.asarray(pr_obj.get("recall", []), dtype=float)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 5.0), dpi=160)
    if len(precision) and len(recall):
        plt.plot(recall, precision, linewidth=2, label="Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.grid(alpha=0.25)
    if len(precision) and len(recall):
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multi_param_trends(df: pd.DataFrame, traj_label_col: str, results_path: Path) -> None:
    sofa_cols = ["sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h"]
    lact_cols = ["lactate_0h", "lactate_6h", "lactate_12h", "lactate_24h", "lactate_48h"]
    t = np.array([0, 6, 12, 24, 48], dtype=float)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.dropna(subset=[traj_label_col]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160, sharex=True)
    ax1, ax2 = axes
    for g in sorted(d[traj_label_col].unique()):
        sub = d[d[traj_label_col] == g]
        gi = int(g)
        c = TRAJ_COLORS[gi] if gi < len(TRAJ_COLORS) else None
        sofa = sub[sofa_cols].to_numpy(dtype=float)
        lact = sub[lact_cols].to_numpy(dtype=float)
        sofa_mean = np.nanmean(sofa, axis=0)
        lact_mean = np.nanmean(lact, axis=0)
        sofa_sem = np.nanstd(sofa, axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        lact_sem = np.nanstd(lact, axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        lbl = f"{_traj_label(g)} (n={len(sub)})"
        ax1.plot(t, sofa_mean, linewidth=2, color=c, label=lbl)
        ax1.fill_between(t, sofa_mean - sofa_sem, sofa_mean + sofa_sem, color=c, alpha=0.18)
        ax2.plot(t, lact_mean, linewidth=2, color=c, label=lbl)
        ax2.fill_between(t, lact_mean - lact_sem, lact_mean + lact_sem, color=c, alpha=0.18)
    ax1.set_title("SOFA trend by trajectory")
    ax1.set_xlabel("Hours")
    ax1.set_ylabel("SOFA")
    ax1.grid(alpha=0.25)
    ax1.legend(title="Trajectory", fontsize=10)
    ax2.set_title("Lactate trend by trajectory")
    ax2.set_xlabel("Hours")
    ax2.set_ylabel("Lactate (mmol/L)")
    ax2.grid(alpha=0.25)
    ax2.legend(title="Trajectory", fontsize=10)
    fig.tight_layout()
    fig.savefig(results_path)
    plt.close(fig)
