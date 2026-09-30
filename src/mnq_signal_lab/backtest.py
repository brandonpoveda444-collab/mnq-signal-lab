from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig, RiskConfig


def _time_ok(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = ts.strftime("%H:%M")
    return start <= t <= end


def simulate(pred: pd.DataFrame, costs: CostConfig, risk: RiskConfig) -> pd.DataFrame:
    rows = []
    daily_pnl: dict[object, float] = {}; daily_count: dict[object, int] = {}
    for row in pred.itertuples():
        day = row.timestamp.date(); sig = int(row.signal)
        allowed = (not risk.kill_switch and sig != 0 and _time_ok(row.timestamp, risk.session_start, risk.session_end)
                   and daily_pnl.get(day, 0.0) > -risk.max_daily_loss_usd
                   and daily_count.get(day, 0) < risk.max_trades_per_day)
        if not allowed:
            continue
        gross_points = sig * (row.future_close - row.close)
        friction_points = costs.spread_ticks * 0.25 + 2 * costs.slippage_ticks_per_side * 0.25
        pnl = (gross_points - friction_points) * risk.point_value * risk.contracts - 2 * costs.commission_per_side * risk.contracts
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        daily_count[day] = daily_count.get(day, 0) + 1
        rows.append({"timestamp": row.timestamp, "direction": sig, "entry": row.close, "exit": row.future_close,
                     "confidence": row.confidence, "correct": int(np.sign(row.future_close-row.close) == sig), "pnl": pnl, "fold": row.fold})
    return pd.DataFrame(rows, columns=["timestamp","direction","entry","exit","confidence","correct","pnl","fold"])


def metrics(pred: pd.DataFrame, trades: pd.DataFrame) -> dict:
    eligible = pred["target"].ne(0)
    signals = pred["signal"].ne(0)
    coverage = float((signals & eligible).sum() / max(eligible.sum(), 1))
    if trades.empty:
        return {"trades": 0, "precision": 0.0, "coverage": coverage, "expectancy_usd": 0.0, "profit_factor": 0.0, "max_drawdown_usd": 0.0, "sharpe_daily": 0.0, "net_pnl_usd": 0.0, "stability_by_fold": {}}
    pnl = trades["pnl"]; equity = pnl.cumsum(); dd = equity - equity.cummax()
    wins = pnl[pnl > 0].sum(); losses = -pnl[pnl < 0].sum()
    daily = trades.assign(day=trades.timestamp.dt.date).groupby("day").pnl.sum()
    sharpe = 0.0 if daily.std(ddof=1) == 0 or len(daily) < 2 else float(np.sqrt(252)*daily.mean()/daily.std(ddof=1))
    folds = trades.groupby("fold").pnl.agg(["count", "mean", "sum"]).round(4).to_dict("index")
    return {"trades": int(len(trades)), "precision": float(trades.correct.mean()), "coverage": coverage,
            "expectancy_usd": float(pnl.mean()), "profit_factor": float(wins/losses) if losses else float("inf"),
            "max_drawdown_usd": float(dd.min()), "sharpe_daily": sharpe, "net_pnl_usd": float(pnl.sum()), "stability_by_fold": folds}


def threshold_tradeoff(pred: pd.DataFrame, costs: CostConfig, risk: RiskConfig, thresholds=(0.50, 0.60, 0.64, 0.70, 0.80)) -> list[dict]:
    """Diagnóstico OOS; no debe usarse para volver a ajustar el test final."""
    rows = []
    original = pred["signal"].copy()
    for threshold in thresholds:
        candidate = pred.copy()
        candidate["signal"] = np.where(candidate["confidence"] >= threshold, candidate["predicted_class"], 0)
        m = metrics(candidate, simulate(candidate, costs, risk))
        rows.append({"threshold": threshold, **{k: m[k] for k in ("trades", "precision", "coverage", "expectancy_usd", "net_pnl_usd", "max_drawdown_usd")}})
    pred["signal"] = original
    return rows
