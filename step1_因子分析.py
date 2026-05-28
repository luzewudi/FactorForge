# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.log_kit import logger
from utils.path_kit import PROJECT_ROOT

# 使用项目统一路径工具定位根目录，保证从任意工作目录启动脚本时都能找到本项目模块。
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from factor_backtest.config_loader import BacktestConfig
from factor_backtest.factor_analysis import run_factor_analysis


def main() -> None:
    """Step1 入口：读取同一份 YAML 配置并执行因子分析。"""
    # Step1：解析命令行参数，获取唯一 YAML 配置文件路径。
    default_config = PROJECT_ROOT_PATH / "config" / "config.yaml"
    parser = argparse.ArgumentParser(description="YAML 驱动的因子分析")
    parser.add_argument("--config", default=default_config, help="YAML 配置路径，默认使用 config/config.yaml")
    args = parser.parse_args()

    # Step2：加载 YAML 配置；因子分析只使用 paths / factors / analysis 模块。
    config = BacktestConfig.load(args.config)

    # Step3：执行因子研究，生成 IC、RankIC、分层净值、统计表和 HTML 报告。
    results = run_factor_analysis(config)

    # Step4：通过项目统一 logger 输出结果位置，避免使用 print。
    logger.debug(f"[step1] 完成 {len(results)} 个因子分析，输出目录：{config.paths.output_folder / 'step1_factor_analysis'}")


if __name__ == "__main__":
    main()
