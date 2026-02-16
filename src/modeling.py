from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from dataclasses import asdict


@dataclass(frozen=True)
class ModelingConfig:
    target_col: str = "mortality_28d"
    id_col: str = "patient_id"
    test_size: float = 0.25
    random_state: int = 42
    model_name: str = "pipeline"


@dataclass(frozen=True)
class ModelOutput:
    model_path: Path
    split_indices_path: Path
    metrics: Dict[str, Any]
    curves: Dict[str, Any]


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_binary_model(
    features_df: pd.DataFrame,
    feature_names: List[str],
    cfg: ModelingConfig,
    models_dir: Path,
    results_dir: Path,
) -> ModelOutput:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in feature_names if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing expected feature columns: {missing}")
    if cfg.target_col not in features_df.columns:
        raise ValueError(f"features_df missing target_col={cfg.target_col}")
    X = features_df[feature_names].copy()
    y = features_df[cfg.target_col].astype(int).to_numpy()
    ids = features_df[cfg.id_col].astype(str).to_numpy() if cfg.id_col in features_df.columns else None
    idx = np.arange(len(features_df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
                random_state=cfg.random_state,
            )),
        ]
    )
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(y_test, y_prob, thr=0.5)
    fpr, tpr, roc_thr = roc_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([], [], [])
    pr, rc, pr_thr = precision_recall_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([], [], [])
    curves = {
        "roc": {"fpr": np.asarray(fpr).tolist(), "tpr": np.asarray(tpr).tolist(), "thr": np.asarray(roc_thr).tolist()},
        "pr": {"precision": np.asarray(pr).tolist(), "recall": np.asarray(rc).tolist(), "thr": np.asarray(pr_thr).tolist()},
    }
    model_path = models_dir / f"{cfg.model_name}.pkl"
    joblib.dump(
        {"pipeline": pipe, "feature_names": feature_names, "config": asdict(cfg)},
        model_path,
    )
    split_indices_path = models_dir / "split_indices.json"
    split_obj: Dict[str, Any] = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
    if ids is not None:
        split_obj["train_patient_id"] = ids[train_idx].tolist()
        split_obj["test_patient_id"] = ids[test_idx].tolist()
    with split_indices_path.open("w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (results_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    return ModelOutput(
        model_path=model_path,
        split_indices_path=split_indices_path,
        metrics=metrics,
        curves=curves,
    )


def plot_roc_pr(model_out: ModelOutput, results_dir: Path, roc_path: Path, pr_path: Path) -> None:
    from src.plots import plot_roc_curve, plot_pr_curve
    results_dir.mkdir(parents=True, exist_ok=True)
    auc = model_out.metrics.get("roc_auc")
    plot_roc_curve(model_out.curves.get("roc", {}), roc_path, auc=auc)
    plot_pr_curve(model_out.curves.get("pr", {}), pr_path)
