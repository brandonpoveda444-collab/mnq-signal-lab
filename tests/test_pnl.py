import pandas as pd

from mnq_signal_lab.backtest import simulate
from mnq_signal_lab.config import CostConfig, RiskConfig


def _pred(signal=1, entry=100.0, exit_=102.0):
    return pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-02 10:00", tz="America/New_York")],
        "signal": [signal], "future_close": [exit_], "close": [entry],
        "confidence": [0.9], "fold": [1],
    })


def test_long_pnl_includes_round_trip_friction():
    # 2 puntos * $2 - (1 spread tick + 2*0.5 slippage ticks)*.25*$2 - $1.24 = $1.76
    trades = simulate(_pred(), CostConfig(), RiskConfig())
    assert len(trades) == 1
    assert trades.pnl.iloc[0] == 1.76


def test_short_pnl_sign():
    trades = simulate(_pred(signal=-1, entry=102, exit_=100), CostConfig(), RiskConfig())
    assert trades.pnl.iloc[0] == 1.76
