from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_roc_curve(roc_obj: Dict[str, Any], save_path: Path) -> None:
    fpr = np.asarray(roc_obj.get("fpr", []), dtype=float)
    tpr = np.asarray(roc_obj.get("tpr", []), dtype=float)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 5.0), dpi=160)
    if len(fpr) and len(tpr):
        plt.plot(fpr, tpr, linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.grid(alpha=0.25)
    if len(fpr) and len(tpr):
        plt.legend()
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
    for g, sub in d.groupby(traj_label_col):
        sofa = sub[sofa_cols].to_numpy(dtype=float)
        lact = sub[lact_cols].to_numpy(dtype=float)
        sofa_mean = np.nanmean(sofa, axis=0)
        lact_mean = np.nanmean(lact, axis=0)
        sofa_sem = np.nanstd(sofa, axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        lact_sem = np.nanstd(lact, axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        ax1.plot(t, sofa_mean, linewidth=2, label=str(g))
        ax1.fill_between(t, sofa_mean - sofa_sem, sofa_mean + sofa_sem, alpha=0.18)
        ax2.plot(t, lact_mean, linewidth=2, label=str(g))
        ax2.fill_between(t, lact_mean - lact_sem, lact_mean + lact_sem, alpha=0.18)
    ax1.set_title("SOFA trend by trajectory")
    ax1.set_xlabel("Hours")
    ax1.set_ylabel("SOFA")
    ax1.grid(alpha=0.25)
    ax1.legend(title="Group")
    ax2.set_title("Lactate trend by trajectory")
    ax2.set_xlabel("Hours")
    ax2.set_ylabel("Lactate (mmol/L)")
    ax2.grid(alpha=0.25)
    ax2.legend(title="Group")
    fig.tight_layout()
    fig.savefig(results_path)
    plt.close(fig)
