from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ensure_synthetic_csv(path: Path, n: int = 200, seed: int = 42) -> None:
    """
    Create a schema-compatible synthetic dataset if it doesn't exist.
    Columns:
      - patient_id
      - SOFA: 0/6/12/24/48h
      - lactate: 0/6/12/24/48h
      - age, sex (0/1)
      - mortality_28d (0/1)
      - time_to_event_days (float)
      - event_observed (0/1)
    """
    if path.exists():
        return
    rng = np.random.default_rng(seed)
    patient_id = np.array([f"P{idx:04d}" for idx in range(n)], dtype=object)
    age = rng.normal(62, 14, size=n).clip(18, 95)
    sex = rng.integers(0, 2, size=n)
    z = rng.choice([0, 1, 2], size=n, p=[0.40, 0.35, 0.25])
    sofa0 = rng.normal(6.5, 2.0, size=n).clip(0, 20)
    slope = np.where(
        z == 0, rng.normal(-0.10, 0.03, n),
        np.where(z == 1, rng.normal(0.01, 0.03, n), rng.normal(0.12, 0.04, n))
    )
    curve = np.where(
        z == 0, rng.normal(0.0015, 0.0008, n),
        np.where(z == 1, rng.normal(0.0003, 0.0008, n), rng.normal(0.0020, 0.0010, n))
    )
    t = np.array([0, 6, 12, 24, 48], dtype=float)
    sofa = []
    for ti in t:
        noise = rng.normal(0, 0.8, n)
        vals = sofa0 + slope * ti + curve * (ti ** 2) + noise
        sofa.append(vals.clip(0, 24))
    sofa = np.vstack(sofa).T
    lact0 = (1.2 + 0.25 * sofa[:, 0] + rng.normal(0, 0.8, n)).clip(0.5, 18.0)
    lact = []
    for i, ti in enumerate(t):
        drift = np.where(z == 0, -0.02 * ti, np.where(z == 1, -0.003 * ti, 0.015 * ti))
        vals = lact0 + 0.05 * (sofa[:, i] - sofa[:, 0]) + drift + rng.normal(0, 0.6, n)
        lact.append(vals.clip(0.5, 25.0))
    lact = np.vstack(lact).T
    risk = (
        -6.0
        + 0.03 * (age - 50)
        + 0.25 * sofa[:, 3]
        + 0.18 * sofa[:, 4]
        + 0.08 * lact[:, 3]
    )
    p_death = _sigmoid(risk)
    mortality_28d = rng.binomial(1, p_death)
    baseline_hazard = 0.03 + 0.015 * p_death
    time_to_event = rng.exponential(1.0 / baseline_hazard)
    time_to_event_days = time_to_event.clip(0.5, 60.0)
    censor_prob = np.where(mortality_28d == 1, 0.05, 0.55)
    censored = rng.binomial(1, censor_prob)
    event_observed = np.where(censored == 1, 0, 1)
    event_observed = np.where(mortality_28d == 1, rng.binomial(1, 0.95, size=n), event_observed)
    df = pd.DataFrame({
        "patient_id": patient_id,
        "sofa_0h": sofa[:, 0],
        "sofa_6h": sofa[:, 1],
        "sofa_12h": sofa[:, 2],
        "sofa_24h": sofa[:, 3],
        "sofa_48h": sofa[:, 4],
        "lactate_0h": lact[:, 0],
        "lactate_6h": lact[:, 1],
        "lactate_12h": lact[:, 2],
        "lactate_24h": lact[:, 3],
        "lactate_48h": lact[:, 4],
        "age": age.round(1),
        "sex": sex.astype(int),
        "mortality_28d": mortality_28d.astype(int),
        "time_to_event_days": time_to_event_days.round(3),
        "event_observed": event_observed.astype(int),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
