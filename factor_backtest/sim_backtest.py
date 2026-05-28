# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config_loader import BacktestConfig, resolve_factor_files
from .data_loader import BacktestData, DateWindow
from .factor_analysis import apply_neutralization, stable_softmax
from .metrics import metrics_to_frame, nav_metrics
from .plot_results import write_simulation_html, write_summary_html
from .simulator import Simulator, vector_to_limit_dict, vector_to_price_dict
from utils.log_kit import logger
from utils.path_kit import PROJECT_ROOT, get_folder_by_root


@dataclass
class SimulationResult:
    factor_name: str
    nav: pd.DataFrame
    trades: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    html_path: Path


def run_simulation(config: BacktestConfig) -> list[SimulationResult]:
    """执行 Step2 真实账户模拟主流程，逐因子输出净值、交易明细和 HTML 报告。"""
    # Step1：加载公共行情数据，并把所有矩阵裁剪到配置中的回测日期区间。
    data = BacktestData(config)
    window = data.date_window(config.analysis.start_date, config.analysis.end_date)
    out_root = Path(get_folder_by_root(config.paths.output_folder, "step2_simulation"))

    # Step2：读取成交价、估值价、交易状态、涨跌停和复利基准净值。
    # simulation.universe 是真实账户模拟的独立股票池，允许和 Step1 因子分析股票池不同。
    universe = data.load_universe(config.simulation.universe, window)
    trade_status = data.load_eod_panel("TradeStatus.npy", window)
    trade_prices, close_prices, limit_status = data.load_simulation_price_panels(config.simulation.trade_price, window)
    benchmark_nav = data.load_benchmark_nav(config.simulation.benchmark, window, compound=True)
    market_cap = data.load_market_cap(window) if config.simulation.weight_method == "market" else None

    results: list[SimulationResult] = []
    summary_rows: list[dict] = []
    factor_files = resolve_factor_files(config, stage="simulation")
    total_factors = len(factor_files)
    logger.info(f"[step2] 共发现 {total_factors} 个因子，开始逐个模拟回测。")
    logger.info(
        f"[step2] 股票池={config.simulation.universe}，选股数量={config.simulation.select_n}，"
        f"调仓周期={config.simulation.rebalance_freq_days}日，加权方法={config.simulation.weight_method}。"
    )
    for factor_idx, factor_path in enumerate(factor_files, start=1):
        started_at = perf_counter()
        factor_name = factor_path.stem
        logger.info(f"[step2][{factor_idx}/{total_factors}] 开始模拟因子：{factor_name}")

        # Step3：因子对齐、股票池过滤、中性化都复用 Step1 的统一逻辑，保证研究和模拟口径一致。
        logger.debug(f"[step2][{factor_idx}/{total_factors}] {factor_name} - Step3：读取、对齐并处理中性化。")
        factor = data.load_factor(factor_path, window)
        factor = np.where(universe, factor, np.nan)
        factor = apply_neutralization(config, data, window, factor_name, factor)
        factor_direction, rankic_mean = load_step1_rankic_direction(config, factor_name)
        direction_text = "选因子值小的股票" if factor_direction else "选因子值大的股票"
        logger.info(
            f"[step2][{factor_idx}/{total_factors}] {factor_name} - "
            f"读取 Step1 RankIC 均值 {rankic_mean:.6f}，自动方向：{direction_text}。"
        )

        logger.debug(f"[step2][{factor_idx}/{total_factors}] {factor_name} - Step4：进入真实账户逐日模拟。")
        result = simulate_one_factor(
            config=config,
            data=data,
            window=window,
            factor_name=factor_name,
            factor=factor,
            universe=universe,
            trade_status=trade_status,
            trade_prices=trade_prices,
            close_prices=close_prices,
            limit_status=limit_status,
            market_cap=market_cap,
            benchmark_nav=benchmark_nav,
            out_root=out_root,
            factor_direction=factor_direction,
            rankic_mean=rankic_mean,
        )
        results.append(result)
        summary_rows.append(build_summary_row(result))
        elapsed = perf_counter() - started_at
        logger.ok(f"[step2][{factor_idx}/{total_factors}] 完成因子：{factor_name}，耗时 {elapsed:.1f} 秒。")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    write_summary_html(summary, out_root / "summary.html", "Step2 Simulation Summary")
    logger.ok(f"[step2] 全部 {total_factors} 个因子模拟完成，汇总报告：{out_root / 'summary.html'}")
    return results


def simulate_one_factor(
    config: BacktestConfig,
    data: BacktestData,
    window: DateWindow,
    factor_name: str,
    factor: np.ndarray,
    universe: np.ndarray,
    trade_status: np.ndarray,
    trade_prices: np.ndarray,
    close_prices: np.ndarray,
    limit_status: np.ndarray,
    market_cap: np.ndarray | None,
    benchmark_nav: pd.Series,
    out_root: Path,
    factor_direction: bool,
    rankic_mean: float,
) -> SimulationResult:
    """模拟单个因子的真实账户回测：上一日因子选股，下一交易日成交并收盘估值。"""
    sim_cfg = config.simulation
    simulator = Simulator(
        initial_cash=sim_cfg.initial_capital,
        commission_rate=sim_cfg.fee,
        stamp_tax_rate=sim_cfg.stamp_duty,
    )
    selections: list[dict] = []
    date_index = window.pandas_index

    for date_idx, date in enumerate(window.dates):
        # 每天先按收盘价估值；若当天是换仓日，再用上一交易日因子生成目标组合。
        turnover = 0.0
        close_dict = vector_to_price_dict(data.tickers, close_prices[:, date_idx])
        if should_rebalance(date_idx, sim_cfg.rebalance_freq_days):
            factor_idx = date_idx - 1
            selected, scores = select_stocks(
                factor=factor[:, factor_idx],
                universe=universe[:, factor_idx],
                trade_status=trade_status[:, date_idx],
                trade_price=trade_prices[:, date_idx],
                tickers=data.tickers,
                select_n=sim_cfg.select_n,
                factor_direction=factor_direction,
            )
            weights = build_target_weights(
                selected=selected,
                scores=scores,
                tickers=data.tickers,
                date_idx=date_idx,
                market_cap=market_cap,
                weight_method=sim_cfg.weight_method,
                factor_direction=factor_direction,
            )
            for rank, stock in enumerate(selected, start=1):
                selections.append(
                    {
                        "date": date,
                        "factor_date": window.dates[factor_idx],
                        "rank": rank,
                        "stock_code": stock,
                        "score": scores.get(stock, np.nan),
                        "target_weight": weights.get(stock, np.nan),
                    }
                )
            trade_dict = vector_to_price_dict(data.tickers, trade_prices[:, date_idx])
            limit_dict = vector_to_limit_dict(data.tickers, limit_status[:, date_idx])
            # 模拟器内部按“先卖后买”执行，并处理滑点、佣金、印花税和涨跌停限制。
            turnover = simulator.adjust_to_target_weights(
                target_weights=weights,
                trade_prices=trade_dict,
                close_prices=close_dict,
                limit_status=limit_dict,
                trade_date=date,
                scores=scores,
                factor_direction=factor_direction,
                slippage=sim_cfg.slippage,
            )
        simulator.mark_to_market(date, close_dict, turnover=turnover)

    # 将模拟器记录整理为净值、交易流水、选股结果和指标表，分别落盘并写入 HTML。
    daily = pd.DataFrame(simulator.daily_records)
    if not daily.empty:
        daily.index = pd.to_datetime(daily["date"], format="%Y%m%d")
    nav_df = pd.DataFrame(index=date_index)
    if not daily.empty:
        nav_df["strategy_nav"] = daily.reindex(date_index)["nav"].astype(float)
        nav_df["cash"] = daily.reindex(date_index)["cash"].astype(float)
        nav_df["position_value"] = daily.reindex(date_index)["position_value"].astype(float)
        nav_df["turnover"] = daily.reindex(date_index)["turnover"].astype(float)
    if benchmark_nav is not None and not benchmark_nav.empty:
        nav_df["benchmark_nav"] = benchmark_nav.reindex(nav_df.index).ffill()
        nav_df["excess_nav"] = nav_df["strategy_nav"] / nav_df["benchmark_nav"]

    trades_df = pd.DataFrame(simulator.trade_records)
    selections_df = pd.DataFrame(selections)
    metrics = nav_metrics(nav_df["strategy_nav"], nav_df.get("benchmark_nav"))
    metrics.update(
        {
            "total_commission": simulator.total_commission,
            "total_stamp_tax": simulator.total_stamp_tax,
            "avg_turnover": float(nav_df["turnover"].mean()) if "turnover" in nav_df else np.nan,
            "trade_count": int(len(trades_df[trades_df["shares"] > 0])) if not trades_df.empty else 0,
            "rankic_mean_from_step1": rankic_mean,
            "auto_factor_direction": "small_is_better" if factor_direction else "large_is_better",
            "simulation_universe": sim_cfg.universe,
            "weight_method": sim_cfg.weight_method,
        }
    )
    metrics_df = metrics_to_frame(metrics)

    factor_out = Path(get_folder_by_root(out_root, factor_name))
    nav_df.to_csv(factor_out / "nav.csv", encoding="utf-8-sig")
    trades_df.to_csv(factor_out / "trades.csv", index=False, encoding="utf-8-sig")
    selections_df.to_csv(factor_out / "selections.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(factor_out / "metrics.csv", index=False, encoding="utf-8-sig")
    html_path = factor_out / f"{factor_name}_simulation.html"
    write_simulation_html(html_path, factor_name, nav_df, metrics_df, trades_df, selections_df)
    maybe_write_quantstats(config, factor_name, nav_df, factor_out)

    return SimulationResult(factor_name, nav_df, trades_df, selections_df, metrics_df, html_path)


def should_rebalance(date_idx: int, freq: int) -> bool:
    """判断当前日期是否为换仓日；第 0 天没有前一日因子，因此不换仓。"""
    if date_idx <= 0:
        return False
    return (date_idx - 1) % freq == 0


def select_stocks(
    factor: np.ndarray,
    universe: np.ndarray,
    trade_status: np.ndarray,
    trade_price: np.ndarray,
    tickers: list[str],
    select_n: float,
    factor_direction: bool,
) -> tuple[list[str], dict[str, float]]:
    """根据当日可交易股票和因子方向生成目标股票列表及对应因子分数。"""
    valid = np.isfinite(factor) & universe & (trade_status == 1) & np.isfinite(trade_price) & (trade_price > 0)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return [], {}
    order = np.argsort(factor[valid_idx], kind="mergesort")
    if not factor_direction:
        order = order[::-1]
    ordered_idx = valid_idx[order]
    count = resolve_selection_count(select_n, len(ordered_idx))
    selected_idx = ordered_idx[:count]
    selected = [tickers[i] for i in selected_idx]
    scores = {tickers[i]: float(factor[i]) for i in selected_idx}
    return selected, scores


def resolve_selection_count(select_n: float, valid_count: int) -> int:
    """解析 select_n：-1 表示全选，0~1 表示比例，>=1 表示固定只数。"""
    if valid_count <= 0:
        return 0
    if select_n == -1:
        return valid_count
    if 0 < select_n < 1:
        return max(1, min(valid_count, int(np.floor(valid_count * select_n))))
    if select_n >= 1:
        return max(1, min(valid_count, int(select_n)))
    raise ValueError("simulation.select_n must be -1, a positive ratio, or a positive count")


def build_target_weights(
    selected: list[str],
    scores: dict[str, float],
    tickers: list[str],
    date_idx: int,
    market_cap: np.ndarray | None,
    weight_method: str,
    factor_direction: bool,
) -> dict[str, float]:
    """按 simulation.weight_method 生成目标持仓权重，供真实账户模拟器下单使用。"""
    if not selected:
        return {}

    method = str(weight_method).strip().lower()
    if method == "equal":
        return equal_target_weights(selected)

    if method == "market":
        return market_target_weights(selected, tickers, date_idx, market_cap)

    if method == "factor_softmax":
        return factor_softmax_target_weights(selected, scores, factor_direction)

    raise ValueError(f"unsupported weight_method: {weight_method}")


def equal_target_weights(selected: list[str]) -> dict[str, float]:
    """等权目标组合：所有入选股票的目标权重相同。"""
    if not selected:
        return {}
    weight = 1.0 / len(selected)
    return {stock: weight for stock in selected}


def market_target_weights(
    selected: list[str],
    tickers: list[str],
    date_idx: int,
    market_cap: np.ndarray | None,
) -> dict[str, float]:
    """市值加权目标组合：按调仓日总市值归一化，市值不可用时回退为等权。"""
    if market_cap is None:
        return equal_target_weights(selected)

    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    values = np.array(
        [
            market_cap[ticker_to_idx[stock], date_idx] if stock in ticker_to_idx else np.nan
            for stock in selected
        ],
        dtype=float,
    )
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return equal_target_weights(selected)

    weights = np.zeros(len(selected), dtype=float)
    weights[valid] = values[valid] / values[valid].sum()
    return {stock: float(weight) for stock, weight in zip(selected, weights)}


def factor_softmax_target_weights(
    selected: list[str],
    scores: dict[str, float],
    factor_direction: bool,
) -> dict[str, float]:
    """因子 softmax 目标组合：先按多头方向调整分数，再 softmax 成目标权重。"""
    raw_scores = np.array([scores.get(stock, np.nan) for stock in selected], dtype=float)
    # factor_direction=True 表示 RankIC 为负、因子值越小越好，因此取负数后再 softmax。
    directional_scores = -raw_scores if factor_direction else raw_scores
    weights = stable_softmax(directional_scores)
    if np.isfinite(weights).all() and weights.sum() > 0:
        return {stock: float(weight) for stock, weight in zip(selected, weights)}
    return equal_target_weights(selected)


def load_step1_rankic_direction(config: BacktestConfig, factor_name: str) -> tuple[bool, float]:
    """从 Step1 统计结果读取 TOTAL RankIC 均值，并据此自动决定 Step2 多头方向。"""
    stats_path = config.paths.output_folder / "step1_factor_analysis" / factor_name / "stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"找不到因子 {factor_name} 的 Step1 统计文件：{stats_path}。"
            f"请先运行：python step1_因子分析.py --config {config.config_path}"
        )

    stats_df = pd.read_csv(stats_path)
    required_cols = {"period", "series", "mean"}
    if not required_cols.issubset(stats_df.columns):
        raise ValueError(f"Step1 统计文件缺少必要列 {required_cols}：{stats_path}")

    mask = (stats_df["period"].astype(str) == "TOTAL") & (stats_df["series"].astype(str) == "RankIC")
    if not mask.any():
        raise ValueError(
            f"Step1 统计文件中找不到 TOTAL RankIC 行：{stats_path}。"
            f"请重新运行对应因子的 Step1。"
        )

    rankic_mean = pd.to_numeric(stats_df.loc[mask, "mean"], errors="coerce").dropna()
    if rankic_mean.empty:
        raise ValueError(
            f"Step1 统计文件中的 TOTAL RankIC mean 无法解析：{stats_path}。"
            f"请重新运行对应因子的 Step1。"
        )

    value = float(rankic_mean.iloc[0])
    # RankIC 均值 >= 0 说明因子值越大未来收益越高，Step2 选大；小于 0 则选小。
    return value < 0, value


def maybe_write_quantstats(config: BacktestConfig, factor_name: str, nav_df: pd.DataFrame, factor_out: Path) -> None:
    """调用项目内置 quantstats 生成增强报告，不再使用外部安装的同名库。"""
    if not config.simulation.enable_quantstats:
        return
    try:
        qs = import_local_quantstats()
    except Exception as exc:
        logger.warning(f"[step2] 本地 quantstats 不可用，跳过 {factor_name} 的增强报告：{exc}")
        return
    if "strategy_nav" not in nav_df or "benchmark_nav" not in nav_df:
        return
    returns = nav_df["strategy_nav"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    benchmark = nav_df["benchmark_nav"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    try:
        qs.reports.html(
            aligned.iloc[:, 0],
            benchmark=aligned.iloc[:, 1],
            output=str(factor_out / "quantstats_report.html"),
            title=f"{factor_name} QuantStats Report",
        )
    except Exception as exc:
        logger.warning(f"[step2] 本地 quantstats 增强报告生成失败，因子 {factor_name}：{exc}")


def import_local_quantstats():
    """只允许导入项目目录下的 quantstats，防止误用 pip/conda 环境中的外部库。"""
    import importlib
    import sys

    project_root = Path(PROJECT_ROOT).resolve()
    local_pkg_dir = project_root / "quantstats"
    if not local_pkg_dir.exists():
        raise FileNotFoundError(f"项目内 quantstats 目录不存在：{local_pkg_dir}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 如果当前进程已经误加载外部 quantstats，先移除缓存，确保后续导入命中本项目目录。
    loaded_qs = sys.modules.get("quantstats")
    if loaded_qs is not None:
        loaded_file = Path(getattr(loaded_qs, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(local_pkg_dir):
            for module_name in list(sys.modules):
                if module_name == "quantstats" or module_name.startswith("quantstats."):
                    del sys.modules[module_name]

    qs = importlib.import_module("quantstats")
    qs_file = Path(qs.__file__).resolve()
    if not qs_file.is_relative_to(local_pkg_dir):
        raise ImportError(f"当前导入的 quantstats 不是项目内版本：{qs_file}")
    logger.debug(f"[step2] 使用项目内本地 quantstats：{qs_file}")
    return qs


def build_summary_row(result: SimulationResult) -> dict:
    """整理单因子的模拟回测摘要行，供 step2 summary.csv/html 使用。"""
    metrics = dict(zip(result.metrics["metric"], result.metrics["value"]))
    row = {"factor": result.factor_name, "html": str(result.html_path)}
    for key in ["total_return", "annual_return", "sharpe", "max_drawdown", "excess_return", "avg_turnover", "trade_count"]:
        if key in metrics:
            row[key] = metrics[key]
    return row
