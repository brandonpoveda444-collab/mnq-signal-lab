from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import ModelConfig


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    fold_boundaries: list[dict]


def _estimator(seed: int):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


def _fit_calibrated(X: pd.DataFrame, y: pd.Series, cfg: ModelConfig):
    cut = int(len(X) * (1 - cfg.calibration_fraction))
    if cut < 100 or len(X) - cut < 50:
        raise ValueError("Insuficientes filas para entrenamiento y calibración temporal")
    base = _estimator(cfg.random_state).fit(X.iloc[:cut], y.iloc[:cut])
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ImportError:  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X.iloc[cut:], y.iloc[cut:])
    return calibrated


def walk_forward_predict(data: pd.DataFrame, feature_cols: list[str], cfg: ModelConfig, min_train: int) -> PredictionResult:
    n = len(data); test_start = int(n * (1 - cfg.test_fraction))
    if test_start < min_train:
        raise ValueError(f"Se requieren al menos {min_train/(1-cfg.test_fraction):.0f} filas útiles")
    test_idx = np.arange(test_start, n)
    chunks = [c for c in np.array_split(test_idx, cfg.n_splits) if len(c)]
    outputs, boundaries = [], []
    for fold, idx in enumerate(chunks, 1):
        train = data.iloc[:idx[0]].copy()
        # Purga muestras cuya etiqueta alcanza el periodo de prueba.
        train = train[train["label_end_time"] < data.iloc[idx[0]]["timestamp"]]
        if len(train) < min_train:
            raise ValueError(f"Fold {fold}: solo {len(train)} filas tras purga")
        model = _fit_calibrated(train[feature_cols], train["target"].astype(int), cfg)
        block = data.iloc[idx].copy()
        proba = model.predict_proba(block[feature_cols])
        classes = model.classes_.astype(int)
        best = proba.argmax(axis=1)
        block["predicted_class"] = classes[best]
        block["confidence"] = proba[np.arange(len(block)), best]
        block["signal"] = np.where(block["confidence"] >= cfg.confidence_threshold, block["predicted_class"], 0).astype(int)
        block["fold"] = fold
        outputs.append(block)
        boundaries.append({"fold": fold, "train_end": str(train.timestamp.iloc[-1]), "test_start": str(block.timestamp.iloc[0]), "test_end": str(block.timestamp.iloc[-1]), "train_rows": len(train), "test_rows": len(block)})
    return PredictionResult(pd.concat(outputs).reset_index(drop=True), boundaries)
