from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib


@dataclass(frozen=True)
class ExplainOutput:
    feature_importance: pd.DataFrame
    saved_path: Optional[Path]


def explain_linear_model(model_path: Path, feature_names: List[str], save_path: Optional[Path] = None) -> ExplainOutput:
    obj: Dict[str, Any] = joblib.load(model_path)
    pipe = obj["pipeline"]
    clf = None
    for _, step in getattr(pipe, "named_steps", {}).items():
        if hasattr(step, "coef_"):
            clf = step
            break
    if clf is None or not hasattr(clf, "coef_"):
        raise ValueError("Could not find linear classifier with coef_ in pipeline.")
    coef = np.asarray(clf.coef_).reshape(-1)
    if len(coef) != len(feature_names):
        raise ValueError(f"coef length {len(coef)} != feature_names length {len(feature_names)}")
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef),
        "sign": np.where(coef >= 0, "+", "-"),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    saved = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        saved = save_path
    return ExplainOutput(feature_importance=df, saved_path=saved)
