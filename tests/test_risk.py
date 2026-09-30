import pandas as pd

from mnq_signal_lab.backtest import simulate
from mnq_signal_lab.config import CostConfig, RiskConfig


def predictions(n=5):
    ts = pd.date_range("2024-01-02 10:00", periods=n, freq="10min", tz="America/New_York")
    return pd.DataFrame({"timestamp": ts, "signal": 1, "future_close": 101.0, "close": 100.0, "confidence": .9, "fold": 1})


def test_max_trades_per_day():
    risk = RiskConfig(max_trades_per_day=2)
    assert len(simulate(predictions(), CostConfig(), risk)) == 2


def test_kill_switch_blocks_everything():
    assert simulate(predictions(), CostConfig(), RiskConfig(kill_switch=True)).empty


def test_session_filter():
    risk = RiskConfig(session_start="11:00", session_end="12:00")
    assert simulate(predictions(), CostConfig(), risk).empty


def test_daily_loss_stops_subsequent_trades():
    p = predictions(4); p["future_close"] = 90.0
    risk = RiskConfig(max_daily_loss_usd=10, max_trades_per_day=9)
    assert len(simulate(p, CostConfig(), risk)) == 1
