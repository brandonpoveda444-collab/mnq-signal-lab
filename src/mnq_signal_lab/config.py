from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"
    csv_path: str | None = None
    databento_dataset: str = "GLBX.MDP3"
    databento_schema: str = "ohlcv-1m"
    databento_symbols: list[str] = field(default_factory=lambda: ["MNQ.FUT"])
    start: str = "2024-01-02"
    end: str = "2024-05-01"
    timezone: str = "America/New_York"
    cache_path: str | None = "work/data_cache.pkl"


@dataclass
class FeatureConfig:
    horizon_bars: int = 5
    label_threshold_ticks: float = 2.0
    tick_size: float = 0.25
    min_train_rows: int = 1500


@dataclass
class ModelConfig:
    confidence_threshold: float = 0.64
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    n_splits: int = 4
    random_state: int = 42


@dataclass
class CostConfig:
    commission_per_side: float = 0.62
    spread_ticks: float = 1.0
    slippage_ticks_per_side: float = 0.5


@dataclass
class RiskConfig:
    max_daily_loss_usd: float = 80.0
    max_trades_per_day: int = 6
    session_start: str = "09:35"
    session_end: str = "15:45"
    flatten_time: str = "15:55"
    point_value: float = 2.0
    contracts: int = 1
    kill_switch: bool = False


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output_dir: str = "outputs/run"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> AppConfig:
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        data=DataConfig(**raw.get("data", {})),
        features=FeatureConfig(**raw.get("features", {})),
        model=ModelConfig(**raw.get("model", {})),
        costs=CostConfig(**raw.get("costs", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        output_dir=raw.get("output_dir", "outputs/run"),
    )
