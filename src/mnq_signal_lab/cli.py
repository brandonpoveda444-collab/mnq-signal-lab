from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Laboratorio MNQ: backtesting/paper trading, sin órdenes reales")
    p.add_argument("run", nargs="?", default="run")
    p.add_argument("--config", help="Ruta YAML; sin ella usa valores seguros y datos sintéticos")
    p.add_argument("--synthetic-periods", type=int, default=12000)
    args = p.parse_args(argv)
    report = run(load_config(args.config), args.synthetic_periods)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
