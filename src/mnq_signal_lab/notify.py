from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


class Notifier:
    def send(self, payload: dict) -> None:
        raise NotImplementedError


class LogNotifier(Notifier):
    def send(self, payload: dict) -> None:
        print(json.dumps({"event": "signal", **payload}, default=str), flush=True)


class WebhookNotifier(Notifier):
    def __init__(self, url: str, timeout: float = 10.0):
        if not url.startswith(("https://", "http://")):
            raise ValueError("Webhook URL inválida")
        self.url, self.timeout = url, timeout

    def send(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Webhook respondió HTTP {response.status}")


@dataclass
class AlertGuard:
    min_interval_seconds: float = 60.0
    _last_key: str | None = None
    _last_sent: float = field(default=-1e30)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if key == self._last_key or now - self._last_sent < self.min_interval_seconds:
            return False
        self._last_key, self._last_sent = key, now
        return True
