# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from factor_backtest.config_loader import BacktestConfig
from factor_backtest.periods import save_period_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate period rebalance npy files from EOD dates.npy")
    parser.add_argument("--config", default=Path("config") / "config.yaml", help="YAML config path")
    parser.add_argument("--eod-path", default=None, help="Override EOD path")
    parser.add_argument("--period-path", default=None, help="Override period output path")
    args = parser.parse_args()

    config = BacktestConfig.load(args.config)
    eod_path = Path(args.eod_path) if args.eod_path else config.paths.eod_path
    period_path = Path(args.period_path) if args.period_path else config.paths.period_path
    names_path, mask_path, dates_path = save_period_files(eod_path=eod_path, period_path=period_path)
    print(f"saved: {names_path}")
    print(f"saved: {mask_path}")
    print(f"saved: {dates_path}")


if __name__ == "__main__":
    main()
