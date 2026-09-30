from mnq_signal_lab.notify import AlertGuard
from mnq_signal_lab.service import latest_signal_payload, render_dashboard
import pandas as pd


def test_alert_guard_deduplicates_and_rate_limits():
    guard = AlertGuard(min_interval_seconds=10)
    assert guard.allow("a", now=0)
    assert not guard.allow("a", now=20)
    assert not guard.allow("b", now=5)
    assert guard.allow("b", now=11)


def test_no_trade_is_silent_and_directional_signal_has_payload():
    neutral = pd.DataFrame({"timestamp": ["2024-01-02T10:00:00"], "signal": [0], "confidence": [.7]})
    _, payload = latest_signal_payload(neutral)
    assert payload is None
    directional = neutral.copy()
    directional.loc[0, "signal"] = 1
    _, payload = latest_signal_payload(directional)
    assert payload["signal"] == "LONG"
    assert payload["paper_only"] is True


def test_dashboard_renders_without_artifacts(tmp_path):
    page = render_dashboard(tmp_path)
    assert "MNQ Signal Lab" in page
    assert "NO TRADE" not in page or "Nunca envía órdenes reales" in page
