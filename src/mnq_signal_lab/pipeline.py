from __future__ import annotations

import json
import time
from pathlib import Path

from .backtest import metrics, simulate, threshold_tradeoff
from .config import AppConfig
from .data import build_continuous_contract, load_data
from .features import FEATURE_COLUMNS, add_labels, make_features
from .model import walk_forward_predict


def run(cfg: AppConfig, synthetic_periods: int = 12000) -> dict:
    started = time.perf_counter()
    raw = build_continuous_contract(load_data(cfg.data, synthetic_periods))
    frame = add_labels(make_features(raw), cfg.features.horizon_bars, cfg.features.label_threshold_ticks, cfg.features.tick_size)
    clean = frame.dropna(subset=FEATURE_COLUMNS + ["target", "label_end_time", "future_close"]).reset_index(drop=True)
    result = walk_forward_predict(clean, FEATURE_COLUMNS, cfg.model, cfg.features.min_train_rows)
    trades = simulate(result.predictions, cfg.costs, cfg.risk)
    report = metrics(result.predictions, trades)
    elapsed = time.perf_counter() - started
    report.update({"data_source": cfg.data.source, "rows_raw": len(raw), "rows_usable": len(clean), "test_rows": len(result.predictions), "folds": result.fold_boundaries,
                   "threshold_tradeoff_oos_diagnostic": threshold_tradeoff(result.predictions, cfg.costs, cfg.risk),
                   "runtime_seconds": round(elapsed, 4), "processed_rows_per_second": round(len(raw) / max(elapsed, 1e-9), 1),
                   "warning": "Resultados sintéticos verifican el software; no validan rentabilidad ni precisión futura." if cfg.data.source == "synthetic" else "Backtest histórico; no garantiza resultados futuros."})
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "config.resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return report
