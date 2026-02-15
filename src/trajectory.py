from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrajectoryConfig:
    timepoints_hours: Tuple[int, ...] = (0, 6, 12, 24, 48)
    sofa_cols: Tuple[str, ...] = ("sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h")
    degree: int = 3
    k_min: int = 2
    k_max: int = 6
    covariance_type: str = "full"
    random_state: int = 42


@dataclass(frozen=True)
class TrajectoryOutput:
    selected_k: int
    label_col: str
    posterior_cols: List[str]
    posterior_df: pd.DataFrame
    coef_df: pd.DataFrame
    bic_table: pd.DataFrame


def _polyfit_per_patient(y: np.ndarray, t: np.ndarray, degree: int) -> np.ndarray:
    return np.polyfit(t, y, deg=degree)


def fit_trajectory_gmm_bic(df: pd.DataFrame, cfg: TrajectoryConfig, results_dir: Path) -> TrajectoryOutput:
    results_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(cfg.timepoints_hours, dtype=float)
    sofa_mat = df.loc[:, list(cfg.sofa_cols)].to_numpy(dtype=float)
    coefs = []
    for i in range(sofa_mat.shape[0]):
        y = sofa_mat[i, :]
        coefs.append(_polyfit_per_patient(y=y, t=t, degree=cfg.degree))
    coefs = np.asarray(coefs)
    coef_cols = [f"poly_c{j}" for j in range(coefs.shape[1])]
    coef_df = pd.DataFrame(coefs, columns=coef_cols)
    coef_df.insert(0, "patient_id", df["patient_id"].astype(str).to_numpy())
    bics = []
    models = {}
    X = coefs
    for k in range(cfg.k_min, cfg.k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cfg.covariance_type,
            random_state=cfg.random_state,
            n_init=10,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append((k, bic))
        models[k] = gmm
    bic_table = pd.DataFrame(bics, columns=["k", "bic"]).sort_values("k")
    selected_k = int(bic_table.loc[bic_table["bic"].idxmin(), "k"])
    best = models[selected_k]
    plt.figure(figsize=(6.0, 4.2), dpi=160)
    plt.plot(bic_table["k"], bic_table["bic"], marker="o", linewidth=2)
    plt.xlabel("Number of classes (K)")
    plt.ylabel("BIC (lower is better)")
    plt.title("BIC selection for GMM trajectory clustering")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(results_dir / "bic_selection.png")
    plt.close()
    probs = best.predict_proba(X)
    labels = best.predict(X)
    posterior_cols = [f"traj_p{i}" for i in range(selected_k)]
    posterior_df = pd.DataFrame(probs, columns=posterior_cols)
    posterior_df.insert(0, "patient_id", df["patient_id"].astype(str).to_numpy())
    label_col = "traj_label"
    posterior_df[label_col] = labels.astype(int)
    return TrajectoryOutput(
        selected_k=selected_k,
        label_col=label_col,
        posterior_cols=posterior_cols,
        posterior_df=posterior_df,
        coef_df=coef_df,
        bic_table=bic_table,
    )


def plot_sofa_trajectories(
    df: pd.DataFrame,
    cfg: TrajectoryConfig,
    traj_label_col: str,
    results_path: Path,
) -> None:
    t = np.asarray(cfg.timepoints_hours, dtype=float)
    sofa_cols = list(cfg.sofa_cols)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.dropna(subset=[traj_label_col]).copy()
    plt.figure(figsize=(7.5, 5.2), dpi=160)
    for g, sub in d.groupby(traj_label_col):
        mat = sub[sofa_cols].to_numpy(dtype=float)
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        plt.plot(t, mean, linewidth=2, label=f"Group {g}")
        plt.fill_between(t, mean - sem, mean + sem, alpha=0.18)
    plt.title("SOFA trajectories by inferred group")
    plt.xlabel("Hours")
    plt.ylabel("SOFA")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path)
    plt.close()


def plot_pca_trajectories(coef_df: pd.DataFrame, label_series: pd.Series, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    d = coef_df.set_index("patient_id").copy()
    labels = label_series.reindex(d.index)
    X = d.to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(6.0, 5.0), dpi=160)
    for g in sorted(labels.dropna().unique()):
        idx = labels == g
        plt.scatter(Z[idx, 0], Z[idx, 1], s=22, alpha=0.75, label=f"Group {g}")
    plt.title("PCA of polynomial coefficients")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path)
    plt.close()
