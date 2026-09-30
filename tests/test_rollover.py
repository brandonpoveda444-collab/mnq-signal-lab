import pandas as pd

from mnq_signal_lab.data import build_continuous_contract


def test_rollover_selects_daily_volume_leader_and_marks_roll():
    ts = pd.to_datetime(["2024-01-02 15:00Z", "2024-01-02 15:00Z", "2024-01-03 15:00Z", "2024-01-03 15:00Z"])
    df = pd.DataFrame({"timestamp": ts, "contract": ["H", "M", "H", "M"], "open": [100,110,101,111], "high":[101,111,102,112], "low":[99,109,100,110], "close":[100,110,101,111], "volume":[100,10,10,100]})
    out = build_continuous_contract(df)
    assert out.contract.tolist() == ["H", "M"]
    assert out.roll.tolist() == [True, True]
    assert abs(out.close.diff().iloc[1]) < 2
