# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config_loader import BacktestConfig, label_period_to_days, resolve_factor_files
from .data_loader import BacktestData, DateWindow
from .metrics import annual_ic_stats, pearson_ic, rank_ic, rolling_t_value, series_stats
from .neutralization import neutralize_factor
from .plot_results import write_factor_analysis_html, write_summary_html
from utils.log_kit import logger
from utils.path_kit import get_folder_by_root


@dataclass
class FactorAnalysisResult:
    factor_name: str
    ic: pd.Series
    rankic: pd.Series
    stats: pd.DataFrame
    nav: pd.DataFrame
    turnover: pd.DataFrame
    offset_summary: pd.DataFrame
    html_path: Path


def run_factor_analysis(config: BacktestConfig) -> list[FactorAnalysisResult]:
    """执行 Step1 因子研究主流程：IC/RankIC、分层净值、换手和 HTML 报告。"""
    # Step1：加载基础行情、日期、股票池等公共数据，并裁剪到配置指定的研究区间。
    data = BacktestData(config)
    analysis = config.analysis
    window = data.date_window(analysis.start_date, analysis.end_date)
    out_root = Path(get_folder_by_root(config.paths.output_folder, "step1_factor_analysis"))

    # Step2：读取 label、股票池、交易状态、周期掩码和 PnL 所需行情；IC 与分层净值分别使用不同数据口径。
    label = data.load_label(analysis.trade_price, analysis.label_days, window)
    period_masks = data.load_period_masks(analysis.period, analysis.offsets, window)
    universe = data.apply_candidate_filters(data.load_universe(analysis.universe, window), analysis.filters, window)
    trade_status = data.load_eod_panel("TradeStatus.npy", window)
    limit_status = np.nan_to_num(data.load_eod_panel("UpDownLimitStatus.npy", window), nan=0.0)
    benchmark = data.load_benchmark_nav(analysis.benchmark, window, compound=False)

    prc, pre_prc, execution_price, adj_factor = data.load_analysis_price_panels(analysis.trade_price, window)
    # Step1 分层权重只有在选择市值加权时才需要额外读取市值矩阵，避免无谓占用内存。
    market_cap = data.load_market_cap(window) if analysis.weight_method == "market" else None

    results: list[FactorAnalysisResult] = []
    summary_rows: list[dict] = []
    factor_files = resolve_factor_files(config, stage="analysis")
    total_factors = len(factor_files)
    label_annualization_days = label_period_to_days(analysis.label_days)
    annualization_factor = np.sqrt(252.0 / label_annualization_days)
    logger.info(f"[step1] 共发现 {total_factors} 个因子，开始逐个计算。")
    logger.info(
        f"[step1] 股票池={analysis.universe}，分组数={analysis.per_divide_num}，"
        f"周期={analysis.period}，offset数={len(period_masks)}，label={analysis.label_days}，"
        f"加权方法={analysis.weight_method}。"
    )
    for factor_idx, factor_path in enumerate(factor_files, start=1):
        started_at = perf_counter()
        factor_name = factor_path.stem
        logger.info(f"[step1][{factor_idx}/{total_factors}] 开始分析因子：{factor_name}")

        # Step3：因子先对齐到 EOD 股票和日期，再按 universe 过滤，并按配置决定是否做中性化。
        logger.debug(f"[step1][{factor_idx}/{total_factors}] {factor_name} - Step3：读取、对齐并处理中性化。")
        factor = data.load_factor(factor_path, window)
        factor = np.where(universe, factor, np.nan)
        factor = apply_neutralization(config, data, window, factor_name, factor)

        # Step4：横截面逐日计算 IC/RankIC，并补充年度、TOTAL 和 RankIC 滚动 t 值统计。
        logger.debug(f"[step1][{factor_idx}/{total_factors}] {factor_name} - Step4：计算 IC、RankIC 和滚动 t 值。")
        ic_series, rankic_series = compute_ic_rankic(factor, label, universe, trade_status, window)
        stats_df = annual_ic_stats(ic_series, rankic_series, annualization_factor=annualization_factor)
        rolling_t = rolling_t_value(
            rankic_series,
            window=analysis.ic_tstat.window,
            min_periods=analysis.ic_tstat.min_periods,
        )
        rankic_mean = series_stats(rankic_series)["mean"]
        if pd.notna(rankic_mean) and rankic_mean < 0:
            rolling_t = -rolling_t

        # Step5：用 t-1 因子在 t 日形成分组权重，再用日度 PnL 公式计算单利分层净值。
        logger.debug(f"[step1][{factor_idx}/{total_factors}] {factor_name} - Step5：计算分组、权重、分层净值和换手。")
        nav_df, turnover_df, offset_nav_df, offset_turnover_df, offset_summary_df = build_multi_offset_layer_outputs(
            factor=factor,
            trade_status=trade_status,
            limit_status=limit_status,
            universe=universe,
            per_divide_num=analysis.per_divide_num,
            period_masks=period_masks,
            weight_method=analysis.weight_method,
            market_cap=market_cap,
            prc=prc,
            pre_prc=pre_prc,
            execution_price=execution_price,
            adj_factor=adj_factor,
            dates=window.pandas_index,
            rankic_mean=rankic_mean,
            benchmark=benchmark,
        )

        # Step6：每个因子独立落盘，便于后续复核明细和直接打开 HTML。
        logger.debug(f"[step1][{factor_idx}/{total_factors}] {factor_name} - Step6：写出明细文件和 HTML 报告。")
        factor_out = Path(get_folder_by_root(out_root, factor_name))
        _write_csv_safely(ic_series.to_frame("IC"), factor_out / "ic.csv")
        _write_csv_safely(rankic_series.to_frame("RankIC"), factor_out / "rankic.csv")
        _write_csv_safely(stats_df, factor_out / "stats.csv", index=False)
        _write_csv_safely(nav_df, factor_out / "nav.csv")
        _write_csv_safely(turnover_df, factor_out / "turnover.csv")
        _write_csv_safely(offset_nav_df, factor_out / "offset_nav.csv")
        _write_csv_safely(offset_turnover_df, factor_out / "offset_turnover.csv")
        _write_csv_safely(offset_summary_df, factor_out / "offset_summary.csv", index=False)

        html_path = factor_out / f"{factor_name}_analysis.html"
        write_factor_analysis_html(
            out_path=html_path,
            factor_name=factor_name,
            ic=ic_series,
            rankic=rankic_series,
            rolling_t=rolling_t,
            nav_df=nav_df,
            turnover_df=turnover_df,
            stats_df=stats_df,
        )
        results.append(
            FactorAnalysisResult(
                factor_name=factor_name,
                ic=ic_series,
                rankic=rankic_series,
                stats=stats_df,
                nav=nav_df,
                turnover=turnover_df,
                offset_summary=offset_summary_df,
                html_path=html_path,
            )
        )
        summary_rows.append(
            build_summary_row(
                factor_name,
                ic_series,
                rankic_series,
                nav_df,
                html_path,
                annualization_factor=annualization_factor,
            )
        )
        elapsed = perf_counter() - started_at
        logger.ok(f"[step1][{factor_idx}/{total_factors}] 完成因子：{factor_name}，耗时 {elapsed:.1f} 秒。")

    summary = pd.DataFrame(summary_rows)
    _write_csv_safely(summary, out_root / "summary.csv", index=False)
    write_summary_html(summary, out_root / "summary.html", "Step1 Factor Analysis Summary")
    logger.ok(f"[step1] 全部 {total_factors} 个因子计算完成，汇总报告：{out_root / 'summary.html'}")
    return results


def _write_csv_safely(df: pd.DataFrame, path: Path, index: bool = True) -> bool:
    """写 CSV；Windows 文件被 Excel/浏览器锁住时不中断整个批处理。"""
    try:
        df.to_csv(path, index=index, encoding="utf-8-sig")
        return True
    except PermissionError:
        logger.warning(f"[step1] 文件被占用，跳过覆盖：{path}")
        return False


def apply_neutralization(
    config: BacktestConfig,
    data: BacktestData,
    window: DateWindow,
    factor_name: str,
    factor: np.ndarray,
) -> np.ndarray:
    """按 analysis.neutralization 配置决定是否做市值或行业市值中性化，并缓存结果。"""
    mode = config.analysis.neutralization
    if mode == "none":
        return factor
    cache_dir = config.paths.output_folder / "neutralized_cache" / mode
    cache_name = f"{factor_name}_{config.analysis.start_date}_{config.analysis.end_date}.npy"
    return neutralize_factor(factor, mode, data, window, cache_dir / cache_name)


def compute_ic_rankic(
    factor: np.ndarray,
    label: np.ndarray,
    universe: np.ndarray,
    trade_status: np.ndarray,
    window: DateWindow,
) -> tuple[pd.Series, pd.Series]:
    """在有效股票池内逐日计算 IC 和 RankIC，输出带日期索引的序列。"""
    valid = universe & np.isfinite(factor) & np.isfinite(label) & (trade_status == 1)
    masked_factor = np.where(valid, factor, np.nan)
    masked_label = np.where(valid, label, np.nan)
    dates = window.pandas_index
    ic = pd.Series(pearson_ic(masked_factor, masked_label), index=dates, name="IC")
    ric = pd.Series(rank_ic(masked_factor, masked_label), index=dates, name="RankIC")
    return ic, ric


def build_multi_offset_layer_outputs(
    factor: np.ndarray,
    trade_status: np.ndarray,
    limit_status: np.ndarray,
    universe: np.ndarray,
    per_divide_num: int,
    period_masks: dict[str, np.ndarray],
    weight_method: str,
    market_cap: np.ndarray | None,
    prc: np.ndarray,
    pre_prc: np.ndarray,
    execution_price: np.ndarray,
    adj_factor: np.ndarray,
    dates: pd.DatetimeIndex,
    rankic_mean: float,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run layer backtests for all offsets and return the ensemble plus offset details."""
    if not period_masks:
        raise ValueError("period_masks is empty")

    group_returns: list[np.ndarray] = []
    turnover_arrays: list[np.ndarray] = []
    offset_nav_frames: list[pd.DataFrame] = []
    offset_turnover_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for period_key, rebalance_mask in period_masks.items():
        groups = create_percentile_groups(
            factor=factor,
            trade_status=trade_status,
            limit_status=limit_status,
            universe=universe,
            per_divide_num=per_divide_num,
            rebalance_mask=rebalance_mask,
        )
        weights = calculate_weights(
            groups=groups,
            method=weight_method,
            market_cap=market_cap,
            factor=factor,
        )
        group_return = calculate_group_daily_return(prc, pre_prc, execution_price, adj_factor, groups, weights)
        nav_df, turnover_df = build_nav_outputs(
            group_return=group_return,
            groups=groups,
            weights=weights,
            dates=dates,
            rankic_mean=rankic_mean,
            benchmark=benchmark,
        )
        group_returns.append(_pad_group_matrix(group_return, per_divide_num, len(dates), fill_value=0.0))
        turnover_arrays.append(
            _pad_group_matrix(calculate_turnover(groups, weights), per_divide_num, max(0, len(dates) - 1), fill_value=np.nan)
        )
        offset_nav_frames.append(nav_df.add_prefix(f"{period_key}__"))
        offset_turnover_frames.append(turnover_df.add_prefix(f"{period_key}__"))
        summary_rows.append(summarize_offset_result(period_key, nav_df, turnover_df))

    ensemble_return = np.nanmean(np.stack(group_returns, axis=0), axis=0)
    ensemble_turnover = np.nanmean(np.stack(turnover_arrays, axis=0), axis=0)
    nav_df = build_nav_from_group_return(ensemble_return, dates, rankic_mean, benchmark)
    turnover_df = turnover_array_to_frame(ensemble_turnover, dates[1:])
    offset_nav_df = pd.concat(offset_nav_frames, axis=1) if offset_nav_frames else pd.DataFrame(index=dates)
    offset_turnover_df = (
        pd.concat(offset_turnover_frames, axis=1) if offset_turnover_frames else pd.DataFrame(index=dates[1:])
    )
    offset_summary_df = pd.DataFrame(summary_rows)
    return nav_df, turnover_df, offset_nav_df, offset_turnover_df, offset_summary_df


def summarize_offset_result(period_key: str, nav_df: pd.DataFrame, turnover_df: pd.DataFrame) -> dict:
    """Summarize one offset sleeve for audit CSV output."""
    avg_turnover = pd.to_numeric(turnover_df.get("average", pd.Series(dtype=float)), errors="coerce")
    active_turnover = avg_turnover[avg_turnover > 0]
    row = {
        "period_key": period_key,
        "average_turnover": float(avg_turnover.mean()) if not avg_turnover.dropna().empty else np.nan,
        "active_average_turnover": float(active_turnover.mean()) if not active_turnover.dropna().empty else np.nan,
        "active_turnover_days": int(active_turnover.count()),
    }
    for col in _group_nav_columns(nav_df):
        row[f"{col}_final_nav"] = float(nav_df[col].iloc[-1]) if not nav_df[col].dropna().empty else np.nan
    if "long_short" in nav_df.columns:
        row["long_short_final_nav"] = float(nav_df["long_short"].iloc[-1])
    return row


def create_percentile_groups(
    factor: np.ndarray,
    trade_status: np.ndarray,
    limit_status: np.ndarray,
    universe: np.ndarray,
    per_divide_num: int,
    turnover: int | None = None,
    rebalance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """按因子从低到高分成 G1..GN；换仓日用 t-1 因子决定 t 日持仓分组。"""
    n_stocks, n_dates = factor.shape
    groups = np.zeros((n_stocks, n_dates), dtype=int)
    factor_masked = np.where(universe, factor, np.nan)
    if rebalance_mask is None:
        if turnover is None:
            raise ValueError("turnover or rebalance_mask is required")
        rebalance_mask = np.array([(date_idx > 0 and date_idx % turnover == 0) for date_idx in range(n_dates)], dtype=bool)
    else:
        rebalance_mask = np.asarray(rebalance_mask, dtype=bool)
        if rebalance_mask.shape[0] != n_dates:
            raise ValueError(f"rebalance_mask length {rebalance_mask.shape[0]} does not match dates {n_dates}")

    for date_idx in range(n_dates):
        if date_idx == 0:
            continue
        if not rebalance_mask[date_idx]:
            groups[:, date_idx] = groups[:, date_idx - 1]
            continue
        fac_col = factor_masked[:, date_idx - 1]
        valid = np.isfinite(fac_col) & (trade_status[:, date_idx] == 1) & (limit_status[:, date_idx] == 0)
        if valid.sum() < per_divide_num:
            continue
        valid_idx = np.where(valid)[0]
        sorted_idx = valid_idx[np.argsort(fac_col[valid_idx], kind="mergesort")]
        for group_id, idx in enumerate(np.array_split(sorted_idx, per_divide_num), start=1):
            groups[idx, date_idx] = group_id
    return groups


def calculate_weights(
    groups: np.ndarray,
    method: str = "equal",
    market_cap: np.ndarray | None = None,
    factor: np.ndarray | None = None,
) -> np.ndarray:
    """根据配置生成组内目标权重：等权、市值加权或因子 softmax 加权。"""
    weights = np.zeros(groups.shape, dtype=float)
    method = str(method).strip().lower()
    max_group = int(groups.max()) if groups.size else 0

    for date_idx in range(groups.shape[1]):
        group_col = groups[:, date_idx]
        for group_id in range(1, max_group + 1):
            mask = group_col == group_id
            if not mask.any():
                continue

            # 等权是最稳健的基础口径，也是市值/因子数据不可用时的兜底方案。
            if method == "equal":
                weights[mask, date_idx] = 1.0 / mask.sum()
                continue

            if method == "market":
                weights[mask, date_idx] = market_cap_group_weights(mask, market_cap, date_idx)
                continue

            if method == "factor_softmax":
                weights[mask, date_idx] = factor_softmax_group_weights(mask, factor, date_idx)
                continue

            raise ValueError(f"unsupported weight_method: {method}")
    return weights


def market_cap_group_weights(mask: np.ndarray, market_cap: np.ndarray | None, date_idx: int) -> np.ndarray:
    """计算单个分组在某一天的市值权重；市值缺失时自动回退为组内等权。"""
    group_size = int(mask.sum())
    out = np.full(group_size, 1.0 / group_size, dtype=float)
    if market_cap is None:
        return out

    cap = market_cap[mask, date_idx]
    valid = np.isfinite(cap) & (cap > 0)
    if valid.any():
        cap_sum = float(cap[valid].sum())
        if np.isfinite(cap_sum) and cap_sum > 0:
            out = np.zeros(group_size, dtype=float)
            out[valid] = cap[valid] / cap_sum
    return out


def factor_softmax_group_weights(mask: np.ndarray, factor: np.ndarray | None, date_idx: int) -> np.ndarray:
    """计算单个分组的因子 softmax 权重；使用上一交易日因子值，尽量贴近真实调仓时点。"""
    group_size = int(mask.sum())
    if factor is None or group_size <= 0:
        return np.full(group_size, 1.0 / max(group_size, 1), dtype=float)

    # 分组本身是 t-1 因子在 t 日形成的，因此 softmax 权重也尽量沿用 t-1 的可见因子值。
    factor_idx = max(date_idx - 1, 0)
    scores = factor[mask, factor_idx]
    weights = stable_softmax(scores)
    if np.isfinite(weights).all() and weights.sum() > 0:
        return weights
    return np.full(group_size, 1.0 / group_size, dtype=float)


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """对一组因子值做数值稳定的 softmax；无有效值时返回全 0 交给上层兜底。"""
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return out

    centered = arr[valid] - np.nanmax(arr[valid])
    exp_value = np.exp(np.clip(centered, -700, 0))
    denom = float(exp_value.sum())
    if np.isfinite(denom) and denom > 0:
        out[valid] = exp_value / denom
    return out


def calculate_group_daily_return(
    prc: np.ndarray,
    pre_prc: np.ndarray,
    execution_price: np.ndarray,
    adj_factor: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """使用原项目日度 PnL 公式计算各分组日收益，输出 group x date 矩阵。"""
    ori_execution = execution_price / adj_factor
    close_rtn = prc / pre_prc - 1.0
    slippage = (ori_execution - prc) / pre_prc
    max_group = int(groups.max()) if groups.size else 0
    returns = np.zeros((max_group, groups.shape[1]), dtype=float)
    weights = np.nan_to_num(weights, nan=0.0)
    for date_idx in range(1, groups.shape[1]):
        for group_id in range(1, max_group + 1):
            wt_prev = np.where(groups[:, date_idx - 1] == group_id, weights[:, date_idx - 1], 0.0)
            wt_curr = np.where(groups[:, date_idx] == group_id, weights[:, date_idx], 0.0)
            trade_amount = wt_curr - wt_prev
            stock_rtn = close_rtn[:, date_idx] * wt_prev - slippage[:, date_idx] * trade_amount
            valid = np.isfinite(stock_rtn)
            returns[group_id - 1, date_idx] = float(stock_rtn[valid].sum()) if valid.any() else 0.0
    return returns


def calculate_turnover(groups: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """计算双边换手率：sum(abs(weight[t+1] - weight[t]))，完全换仓约等于 2。"""
    max_group = int(groups.max()) if groups.size else 0
    out = np.full((max_group, max(0, groups.shape[1] - 1)), np.nan, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0)
    for period_idx in range(groups.shape[1] - 1):
        for group_id in range(1, max_group + 1):
            wt_t = np.where(groups[:, period_idx] == group_id, weights[:, period_idx], 0.0)
            wt_t1 = np.where(groups[:, period_idx + 1] == group_id, weights[:, period_idx + 1], 0.0)
            out[group_id - 1, period_idx] = float(np.abs(wt_t1 - wt_t).sum())
    return out


def build_nav_outputs(
    group_return: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    rankic_mean: float,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把分组日收益转换为单利净值，并按 RankIC 方向生成方向调整多空净值。"""
    nav = build_nav_from_group_return(group_return, dates, rankic_mean, benchmark)
    turnover = calculate_turnover(groups, weights)
    turnover_df = turnover_array_to_frame(turnover, dates[1:])
    return nav, turnover_df


def build_nav_from_group_return(
    group_return: np.ndarray,
    dates: pd.DatetimeIndex,
    rankic_mean: float,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Convert group daily returns into simple NAV curves."""
    nav = pd.DataFrame(index=dates)
    for i in range(group_return.shape[0]):
        nav[f"G{i + 1}"] = 1.0 + np.nan_to_num(group_return[i], nan=0.0).cumsum()
    if group_return.shape[0] >= 2:
        if pd.isna(rankic_mean) or rankic_mean >= 0:
            ls_ret = group_return[-1] - group_return[0]
        else:
            ls_ret = group_return[0] - group_return[-1]
        nav["long_short"] = 1.0 + np.nan_to_num(ls_ret, nan=0.0).cumsum()
    if benchmark is not None and not benchmark.empty:
        nav["benchmark"] = benchmark.reindex(nav.index).ffill()
    return nav


def turnover_array_to_frame(turnover: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert a group x date turnover array into the standard report frame."""
    index = dates[: turnover.shape[1]] if len(dates) != turnover.shape[1] else dates
    turnover_df = pd.DataFrame(index=index)
    for i in range(turnover.shape[0]):
        turnover_df[f"G{i + 1}"] = turnover[i, : len(index)]
    if not turnover_df.empty:
        turnover_df["average"] = turnover_df.mean(axis=1)
    return turnover_df


def _pad_group_matrix(matrix: np.ndarray, group_count: int, date_count: int, fill_value: float) -> np.ndarray:
    """Pad group matrices so all offsets can be stacked safely."""
    out = np.full((group_count, date_count), fill_value, dtype=float)
    rows = min(group_count, matrix.shape[0])
    cols = min(date_count, matrix.shape[1])
    if rows > 0 and cols > 0:
        out[:rows, :cols] = matrix[:rows, :cols]
    return out


def _group_nav_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("G") and str(col)[1:].isdigit()]
    return sorted(cols, key=lambda col: int(str(col)[1:]))


def build_summary_row(
    factor_name: str,
    ic: pd.Series,
    rankic: pd.Series,
    nav: pd.DataFrame,
    html_path: Path,
    annualization_factor: float = 1.0,
) -> dict:
    """整理单因子的摘要行，供 step1 summary.csv/html 使用。"""
    ic_stats = series_stats(ic, annualization_factor=annualization_factor)
    rankic_stats = series_stats(rankic, annualization_factor=annualization_factor)
    row = {
        "factor": factor_name,
        "IC_mean": ic_stats["mean"],
        "ICIR": ic_stats["ir"],
        "Annualized_ICIR": ic_stats["annualized_ir"],
        "IC_t": ic_stats["t_value"],
        "IC_p": ic_stats["p_value"],
        "RankIC_mean": rankic_stats["mean"],
        "RankICIR": rankic_stats["ir"],
        "Annualized_RankICIR": rankic_stats["annualized_ir"],
        "RankIC_t": rankic_stats["t_value"],
        "RankIC_p": rankic_stats["p_value"],
        "html": str(html_path),
    }
    if "long_short" in nav.columns:
        row["long_short_final_nav"] = float(nav["long_short"].iloc[-1])
    return row
