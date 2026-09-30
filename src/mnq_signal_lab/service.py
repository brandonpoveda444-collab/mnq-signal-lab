from __future__ import annotations

import json
import os
import signal
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from .config import load_config
from .notify import AlertGuard, LogNotifier, WebhookNotifier
from .pipeline import run

STATE = {"healthy": True, "last_success": None, "last_error": None, "output_dir": "outputs/run"}


def latest_signal_payload(predictions: pd.DataFrame) -> tuple[str, dict | None]:
    """Devuelve una alerta solo para una señal operable; NO_TRADE queda silenciada."""
    latest = predictions.iloc[-1]
    signal_name = {1: "LONG", -1: "SHORT", 0: "NO_TRADE"}[int(latest.signal)]
    if signal_name == "NO_TRADE":
        return f"{latest.timestamp}:{signal_name}", None
    return f"{latest.timestamp}:{signal_name}", {
        "type": "paper_signal", "paper_only": True, "timestamp": latest.timestamp,
        "signal": signal_name, "confidence": float(latest.confidence),
    }


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(STATE).encode()
            status, content_type = (200 if STATE["healthy"] else 503), "application/json"
        elif self.path == "/":
            body = render_dashboard(Path(str(STATE["output_dir"]))).encode()
            status, content_type = 200, "text/html; charset=utf-8"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *_):
        return


def render_dashboard(output_dir: Path) -> str:
    metrics_path, predictions_path = output_dir / "metrics.json", output_dir / "predictions.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    latest = None
    if predictions_path.exists():
        frame = pd.read_csv(predictions_path)
        if not frame.empty:
            latest = frame.iloc[-1]
    signal = "SIN DATOS" if latest is None else {1: "LONG", -1: "SHORT", 0: "NO TRADE"}.get(int(latest.signal), "NO TRADE")
    confidence = 0.0 if latest is None else float(latest.confidence)
    signal_class = {"LONG": "long", "SHORT": "short", "NO TRADE": "neutral"}.get(signal, "neutral")
    tradeoff = metrics.get("threshold_tradeoff_oos_diagnostic", [])
    rows = "".join(
        f"<tr><td>{r['threshold']:.2f}</td><td>{r['trades']}</td><td>{r['precision']:.1%}</td>"
        f"<td>{r['coverage']:.1%}</td><td>${r['expectancy_usd']:.2f}</td><td>${r['max_drawdown_usd']:.2f}</td></tr>"
        for r in tradeoff
    ) or "<tr><td colspan='6'>Aún no hay resultados.</td></tr>"
    def metric(key, fmt="{}"):
        return fmt.format(metrics.get(key, 0))
    warning = escape(str(metrics.get("warning", "Esperando la primera ejecución.")))
    return f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>MNQ Signal Lab</title><style>
:root{{--bg:#07111f;--panel:#101d30;--muted:#91a4bc;--text:#edf4ff;--accent:#55d6be;--red:#ff6b7a;--amber:#ffc857}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07111f,#0b1829);color:var(--text);font:15px system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:center}}
h1{{margin:0;font-size:30px}}.subtitle,.label{{color:var(--muted)}}.badge{{padding:10px 14px;border:1px solid #29405d;border-radius:999px}}
.warning{{margin:22px 0;padding:15px;border-left:4px solid var(--amber);background:#2b2516;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid #20344d;border-radius:14px;padding:18px}}
.value{{font-size:27px;font-weight:750;margin-top:8px}}.signal{{font-size:38px}}.long{{color:var(--accent)}}.short{{color:var(--red)}}.neutral{{color:var(--amber)}}
section{{margin-top:24px}}table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:14px;overflow:hidden}}th,td{{padding:12px;text-align:right;border-bottom:1px solid #20344d}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted)}}
.note{{color:var(--muted);line-height:1.55}}@media(max-width:650px){{main{{padding:16px}}header{{display:block}}}}
</style></head><body><main><header><div><h1>MNQ Signal Lab</h1><div class='subtitle'>Investigación, backtesting y paper trading · Nunca envía órdenes reales</div></div><div class='badge'>Estado: {'SALUDABLE' if STATE['healthy'] else 'ERROR'}</div></header>
<div class='warning'>{warning}</div><div class='grid'>
<div class='card'><div class='label'>Señal más reciente</div><div class='value signal {signal_class}'>{signal}</div><div class='note'>Confianza calibrada: {confidence:.1%}</div></div>
<div class='card'><div class='label'>Operaciones OOS</div><div class='value'>{metric('trades')}</div><div class='note'>Con límites de riesgo y costes</div></div>
<div class='card'><div class='label'>Precisión por señal</div><div class='value'>{metric('precision','{:.1%}')}</div><div class='note'>Aciertos entre señales ejecutadas</div></div>
<div class='card'><div class='label'>Cobertura</div><div class='value'>{metric('coverage','{:.1%}')}</div><div class='note'>Oportunidades con señal</div></div>
<div class='card'><div class='label'>Expectativa neta</div><div class='value'>${metric('expectancy_usd','{:.2f}')}</div><div class='note'>Promedio por operación, tras costes</div></div>
<div class='card'><div class='label'>PnL neto simulado</div><div class='value'>${metric('net_pnl_usd','{:.2f}')}</div><div class='note'>No es rendimiento real</div></div>
<div class='card'><div class='label'>Drawdown máximo</div><div class='value'>${metric('max_drawdown_usd','{:.2f}')}</div><div class='note'>Caída pico-valle del backtest</div></div>
<div class='card'><div class='label'>Rendimiento</div><div class='value'>{metric('processed_rows_per_second','{:,.0f}')} filas/s</div><div class='note'>{metric('runtime_seconds','{:.2f}')} segundos</div></div></div>
<section><h2>Equilibrio entre confianza y actividad</h2><p class='note'>Umbrales altos producen menos señales. La selección final debe hacerse en desarrollo y evaluarse una sola vez en un test sellado.</p>
<table><thead><tr><th>Umbral</th><th>Trades</th><th>Precisión</th><th>Cobertura</th><th>Expectativa</th><th>Drawdown</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Cómo leer el panel</h2><p class='note'><b>NO TRADE</b> es una decisión válida: el modelo se abstiene si su probabilidad no supera el umbral. Precisión sin cobertura puede ser engañosa, y ninguna métrica sintética valida rentabilidad. El servicio está limitado a un contrato MNQ simulado y conserva kill switch, pérdida diaria máxima, horario y máximo de operaciones.</p></section>
</main></body></html>"""


def _health_server(port: int):
    ThreadingHTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


def main() -> int:
    config_path = os.getenv("MNQ_CONFIG", "config.example.yaml")
    interval = max(30, int(os.getenv("MNQ_POLL_SECONDS", "300")))
    guard = AlertGuard(float(os.getenv("MNQ_ALERT_MIN_INTERVAL", "60")))
    notifier = WebhookNotifier(os.environ["MNQ_WEBHOOK_URL"]) if os.getenv("MNQ_WEBHOOK_URL") else LogNotifier()
    STATE["output_dir"] = load_config(config_path).output_dir
    threading.Thread(target=_health_server, args=(int(os.getenv("PORT", "8080")),), daemon=True).start()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    failures = 0
    while not stopped:
        try:
            report = run(load_config(config_path))
            STATE.update(healthy=True, last_success=time.time(), last_error=None)
            cfg = load_config(config_path)
            predictions = pd.read_csv(Path(cfg.output_dir) / "predictions.csv")
            key, payload = latest_signal_payload(predictions)
            if payload is not None and guard.allow(key):
                notifier.send(payload)
            failures = 0
            deadline = time.monotonic() + interval
        except Exception as exc:
            failures += 1
            STATE.update(healthy=False, last_error=str(exc))
            print(json.dumps({"event": "service_error", "attempt": failures, "error": str(exc)}), flush=True)
            deadline = time.monotonic() + min(300, 2 ** min(failures, 8))
        while not stopped and time.monotonic() < deadline:
            time.sleep(min(1, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
