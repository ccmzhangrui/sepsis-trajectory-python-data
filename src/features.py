from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    timepoints_hours: Tuple[int, ...] = (0, 6, 12, 24, 48)
    sofa_cols: Tuple[str, ...] = ("sofa_0h", "sofa_6h", "sofa_12h", "sofa_24h", "sofa_48h")
    lactate_cols: Tuple[str, ...] = ("lactate_0h", "lactate_6h", "lactate_12h", "lactate_24h", "lactate_48h")
    include_entropy: bool = True
    include_spectral: bool = True
    random_state: int = 42


def _safe_diff(a: np.ndarray, i: int, j: int) -> np.ndarray:
    return a[:, j] - a[:, i]


def _approx_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < (m + 2):
        return float("nan")

    def _phi(mm: int) -> float:
        N = x.size - mm + 1
        X = np.array([x[i:i+mm] for i in range(N)])
        C = np.sum(np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2) <= (r * np.std(x) + 1e-12), axis=0) / N
        return float(np.sum(np.log(C + 1e-12)) / N)
    return _phi(m) - _phi(m + 1)


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < (m + 2):
        return float("nan")
    sd = np.std(x) + 1e-12
    tol = r * sd

    def _count(mm: int) -> int:
        N = x.size - mm + 1
        X = np.array([x[i:i+mm] for i in range(N)])
        cnt = 0
        for i in range(N):
            dist = np.max(np.abs(X - X[i]), axis=1)
            cnt += int(np.sum(dist <= tol) - 1)
        return cnt
    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return float("nan")
    return float(-np.log((A + 1e-12) / (B + 1e-12)))


def _fft_bandpower(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    return float(np.sum(spec[1:]))


def build_feature_table(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    traj_p_cols: List[str],
    traj_label_col: str,
) -> tuple:
    t = np.asarray(cfg.timepoints_hours, dtype=float)
    sofa = df.loc[:, list(cfg.sofa_cols)].to_numpy(dtype=float)
    lact = df.loc[:, list(cfg.lactate_cols)].to_numpy(dtype=float)
    out = pd.DataFrame({
        "patient_id": df["patient_id"].astype(str).to_numpy(),
        "mortality_28d": df["mortality_28d"].astype(int).to_numpy(),
        "age": pd.to_numeric(df["age"], errors="coerce").to_numpy(),
        "sex": df["sex"].astype(int).to_numpy(),
        traj_label_col: df[traj_label_col].astype(int).to_numpy() if traj_label_col in df.columns else np.nan,
    })
    for c in traj_p_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    out["sofa_mean"] = np.nanmean(sofa, axis=1)
    out["sofa_std"] = np.nanstd(sofa, axis=1)
    out["sofa_delta_0_24"] = _safe_diff(sofa, 0, 3)
    out["sofa_delta_0_48"] = _safe_diff(sofa, 0, 4)
    out["sofa_slope_0_48"] = (sofa[:, 4] - sofa[:, 0]) / (t[4] - t[0] + 1e-12)
    out["lact_mean"] = np.nanmean(lact, axis=1)
    out["lact_std"] = np.nanstd(lact, axis=1)
    out["lact_delta_0_24"] = _safe_diff(lact, 0, 3)
    out["lact_delta_0_48"] = _safe_diff(lact, 0, 4)
    out["lact_clear_0_24"] = (lact[:, 0] - lact[:, 3]) / (lact[:, 0] + 1e-6)
    out["lact_clear_0_48"] = (lact[:, 0] - lact[:, 4]) / (lact[:, 0] + 1e-6)
    if cfg.include_entropy:
        out["sofa_apen"] = [_approx_entropy(sofa[i, :]) for i in range(sofa.shape[0])]
        out["sofa_sampen"] = [_sample_entropy(sofa[i, :]) for i in range(sofa.shape[0])]
        out["lact_apen"] = [_approx_entropy(lact[i, :]) for i in range(lact.shape[0])]
        out["lact_sampen"] = [_sample_entropy(lact[i, :]) for i in range(lact.shape[0])]
    if cfg.include_spectral:
        out["sofa_bandpower"] = [_fft_bandpower(sofa[i, :]) for i in range(sofa.shape[0])]
        out["lact_bandpower"] = [_fft_bandpower(lact[i, :]) for i in range(lact.shape[0])]
    feature_names: List[str] = []
    feature_names += ["age", "sex"]
    feature_names += traj_p_cols
    feature_names += [
        "sofa_mean", "sofa_std", "sofa_delta_0_24", "sofa_delta_0_48", "sofa_slope_0_48",
        "lact_mean", "lact_std", "lact_delta_0_24", "lact_delta_0_48",
        "lact_clear_0_24", "lact_clear_0_48",
    ]
    if cfg.include_entropy:
        feature_names += ["sofa_apen", "sofa_sampen", "lact_apen", "lact_sampen"]
    if cfg.include_spectral:
        feature_names += ["sofa_bandpower", "lact_bandpower"]
    return out, feature_names
