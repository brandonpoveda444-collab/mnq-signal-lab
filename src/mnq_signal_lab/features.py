from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_15", "vol_15", "vol_60", "range_1", "atr_14",
    "trend_10_30", "volume_z_30", "minute_sin", "minute_cos", "rth", "spread_proxy",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Todas las variables en t usan únicamente observaciones disponibles hasta t."""
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    logc = np.log(x["close"])
    x["ret_1"] = logc.diff()
    x["ret_5"] = logc.diff(5)
    x["ret_15"] = logc.diff(15)
    x["vol_15"] = x["ret_1"].rolling(15).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()
    x["range_1"] = (x["high"] - x["low"]) / x["close"]
    prev = x["close"].shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr_14"] = tr.rolling(14).mean() / x["close"]
    x["trend_10_30"] = x["close"].rolling(10).mean() / x["close"].rolling(30).mean() - 1
    vm = x["volume"].rolling(30).mean(); vs = x["volume"].rolling(30).std()
    x["volume_z_30"] = (x["volume"] - vm) / vs.replace(0, np.nan)
    minute = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
    x["minute_sin"] = np.sin(2*np.pi*minute/1440)
    x["minute_cos"] = np.cos(2*np.pi*minute/1440)
    x["rth"] = ((minute >= 570) & (minute <= 960)).astype(float)
    x["spread_proxy"] = (x["high"] - x["low"]) / np.sqrt(x["volume"].clip(lower=1)) / x["close"]
    return x


def add_labels(df: pd.DataFrame, horizon: int, threshold_ticks: float, tick_size: float) -> pd.DataFrame:
    x = df.copy()
    x["future_close"] = x["close"].shift(-horizon)
    move = x["future_close"] - x["close"]
    threshold = threshold_ticks * tick_size
    x["target"] = np.select([move > threshold, move < -threshold], [1, -1], default=0).astype(int)
    x.loc[x["future_close"].isna(), "target"] = np.nan
    x["label_end_time"] = x["timestamp"].shift(-horizon)
    return x
