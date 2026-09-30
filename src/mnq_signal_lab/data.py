from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

REQUIRED = {"timestamp", "open", "high", "low", "close", "volume"}


def _normalize(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = ts.dt.tz_convert(timezone)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if "symbol" not in out:
        out["symbol"] = "MNQ_SYNTH"
    return out


def generate_synthetic(start: str, periods: int = 12000, seed: int = 7, timezone: str = "America/New_York") -> pd.DataFrame:
    """Serie reproducible con regímenes; sirve para ejecución, no para validar alfa."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="min", tz=timezone)
    regime = np.repeat([0.00002, -0.000015, 0.0, 0.00001], periods // 4 + 1)[:periods]
    eps = rng.normal(0, 0.00032, periods)
    ret = regime + eps + 0.12 * np.r_[0.0, eps[:-1]]
    close = 17000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    width = np.abs(rng.normal(1.8, 0.7, periods))
    high = np.maximum(open_, close) + width
    low = np.minimum(open_, close) - width
    volume = rng.lognormal(6.0, 0.45, periods).astype(int)
    df = pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "symbol": "MNQ_SYNTH"})
    return df


def load_csv(path: str, timezone: str) -> pd.DataFrame:
    return _normalize(pd.read_csv(Path(path)), timezone)


def load_databento(cfg: DataConfig) -> pd.DataFrame:
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("Instala el extra 'databento' para usar esta fuente") from exc
    if not os.getenv("DATABENTO_API_KEY"):
        raise RuntimeError("Falta DATABENTO_API_KEY")
    client = db.Historical()
    store = client.timeseries.get_range(
        dataset=cfg.databento_dataset, schema=cfg.databento_schema,
        symbols=cfg.databento_symbols, stype_in="continuous", start=cfg.start, end=cfg.end,
    )
    df = store.to_df().reset_index().rename(columns={"ts_event": "timestamp", "instrument_id": "symbol"})
    return _normalize(df, cfg.timezone)


def load_data(cfg: DataConfig, synthetic_periods: int = 12000) -> pd.DataFrame:
    cache = Path(cfg.cache_path) if cfg.cache_path else None
    # La fuente sintética se regenera para que periods/seed sean inequívocos.
    if cache and cache.exists() and cfg.source != "synthetic":
        return pd.read_pickle(cache)
    if cfg.source == "synthetic":
        return generate_synthetic(cfg.start, synthetic_periods, timezone=cfg.timezone)
    if cfg.source == "csv":
        if not cfg.csv_path:
            raise ValueError("data.csv_path es obligatorio con source=csv")
        result = load_csv(cfg.csv_path, cfg.timezone)
    elif cfg.source == "databento":
        result = load_databento(cfg)
    else:
        raise ValueError(f"Fuente desconocida: {cfg.source}")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_pickle(cache)
    return result


def build_continuous_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona por día el contrato con mayor volumen y ajusta gaps hacia atrás."""
    if "contract" not in df.columns:
        result = df.copy()
        result["roll"] = False
        return result
    x = df.copy().sort_values("timestamp")
    x["trade_date"] = x["timestamp"].dt.date
    vols = x.groupby(["trade_date", "contract"], observed=True)["volume"].sum()
    leaders = vols.groupby(level=0).idxmax().map(lambda z: z[1])
    x = x.join(leaders.rename("active_contract"), on="trade_date")
    x = x[x["contract"] == x["active_contract"]].copy()
    x["roll"] = x["contract"].ne(x["contract"].shift())
    adjustment = 0.0
    adj = np.zeros(len(x))
    for i in range(1, len(x)):
        if x["roll"].iloc[i]:
            adjustment += x["close"].iloc[i - 1] - x["open"].iloc[i]
        adj[i] = adjustment
    for col in ["open", "high", "low", "close"]:
        x[f"raw_{col}"] = x[col]
        x[col] = x[col] + adj
    return x.drop(columns=["trade_date", "active_contract"]).reset_index(drop=True)
