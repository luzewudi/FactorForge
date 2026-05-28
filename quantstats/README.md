# QuantStats 量化投资组合分析工具

> 本项目是基于 [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) 的中文本地化版本，
> 专为中国量化投资者和分析师优化，所有文档、注释和报告输出均已汉化。

---

## 项目结构

```
quantstats/
├── __init__.py              # 包入口，导出公共 API，提供 extend_pandas() 方法
│
├── stats.py                  # 统计分析核心（60+ 函数）
│   ├── 收益指标      夏普比率、索提诺比率、卡尔玛比率、CAGR
│   ├── 风险指标      最大回撤、波动率、Beta、VaR、CVaR、Ulcer Index
│   ├── 绩效指标      胜率、平均收益/亏损、连续盈亏、Gain/Pain Ratio
│   └── 比较函数      Alpha、Beta、相关性、信息比率、特雷诺比率
│
├── reports.py                # 报告生成
│   ├── html()       生成 HTML tearsheet 报告（支持中文）
│   ├── full()       全面绩效分析报告
│   ├── basic()      基础绩效报告
│   ├── metrics()    计算并显示绩效指标表格
│   └── plots()      生成全套可视化图表
│
├── plots.py                  # 可视化入口（导出 _plotting/wrappers.py）
│
├── utils.py                  # 工具函数
│   ├── validate_input()       输入验证
│   ├── to_returns()          价格转收益率
│   ├── to_prices()           收益率转价格
│   ├── aggregate_returns()   收益率聚合（日→月→年）
│   └── group_returns()       按时间分组聚合
│
├── _plotting/                # 可视化子包
│   ├── core.py               # 底层 matplotlib 绘图函数
│   │   ├── plot_returns_bars()         年度收益柱状图
│   │   ├── plot_timeseries()           时间序列折线图
│   │   ├── plot_histogram()            收益分布直方图
│   │   ├── plot_rolling_stats()       滚动统计指标
│   │   ├── plot_rolling_beta()         滚动贝塔
│   │   ├── plot_longest_drawdowns()    最长回撤期间
│   │   ├── plot_distribution()          分位数箱线图
│   │   └── plot_table()                数据表格
│   │
│   └── wrappers.py            # 高级绘图 API
│       ├── snapshot()               投资组合综合快照
│       ├── returns()                累计收益率
│       ├── log_returns()            对数收益率
│       ├── daily_returns()          日收益率
│       ├── yearly_returns()         年度收益
│       ├── histogram()             直方图
│       ├── drawdown()              回撤图
│       ├── drawdowns_periods()    回撤期间
│       ├── monthly_heatmap()       月度热力图
│       ├── rolling_volatility()    滚动波动率
│       ├── rolling_sharpe()        滚动夏普
│       └── rolling_sortino()       滚动索提诺
│
├── _compat.py                 # pandas/numpy 兼容性处理
│   └── safe_resample()        兼容 pandas 2.2.0+ 频率别名（M→ME, Q→QE 等）
│
├── _numpy_compat.py           # NumPy 版本兼容
│   └── 处理 np.product→np.prod 等已弃用 API
│
├── _zh.py                     # 中文本地化
│   ├── zh_columns()           指标名称中文化
│   ├── zh_index()             索引中文化
│   ├── zh_html()              HTML 中文化
│   ├── zh_metric()            指标中文化
│   └── zh_text()              通用文本中文化
│
└── report.html                # HTML 报告模板
```

---

## 快速开始

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn tabulate
pip install scipy  # 用于部分统计函数
```

### 基本使用

```python
import pandas as pd
import quantstats as qs

# 读取本地收益率数据（支持 pandas Series 和 DataFrame）
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)['strategy']
benchmark = pd.read_csv('benchmark.csv', index_col=0, parse_dates=True)['benchmark']

# 方法 1：直接调用统计函数
sharpe = qs.stats.sharpe(returns)
sortino = qs.stats.sortino(returns)
max_dd = qs.stats.max_drawdown(returns)

# 方法 2：通过 extend_pandas 扩展 pandas
qs.extend_pandas()
returns.sharpe()       # 直接在 Series 上调用
returns.sortino()
returns.max_drawdown()

# 方法 3：完整报告
qs.reports.full(returns, benchmark=benchmark)

# 方法 4：生成 HTML 报告
qs.reports.html(returns, benchmark=benchmark, output='report.html')
```

---

## 核心 API 参考

### 统计分析 (stats.py)

| 函数 | 说明 |
|------|------|
| `sharpe(returns, rf=0, periods=252)` | 夏普比率 |
| `sortino(returns, rf=0, periods=252)` | 索提诺比率 |
| `max_drawdown(returns)` | 最大回撤 |
| `cagr(returns, periods=252)` | 年化增长率 (CAGR) |
| `volatility(returns, periods=252)` | 年化波动率 |
| `calmar(returns)` | 卡尔玛比率 |
| `omega(returns, required_return=0)` | Omega 比率 |
| `tail_ratio(returns)` | 尾部比率 |
| `value_at_risk(returns, sigma=1)` | VaR（风险价值） |
| `conditional_value_at_risk(returns)` | CVaR（条件风险价值） |
| `greeks(returns, benchmark)` | 贝塔和阿尔法 |
| `compare(returns, benchmark, aggregate)` | 与基准比较 |
| `to_drawdown_series(returns)` | 回撤序列 |
| `drawdown_details(dd)` | 回撤详情 |

### 可视化 (plots.py)

| 函数 | 说明 |
|------|------|
| `snapshot(returns)` | 投资组合综合快照（三面板） |
| `returns(returns, benchmark)` | 累计收益率 |
| `yearly_returns(returns, benchmark)` | 年度收益柱状图 |
| `monthly_heatmap(returns, benchmark)` | 月度收益热力图 |
| `drawdown(returns)` | 回撤图 |
| `histogram(returns, benchmark)` | 收益分布直方图 |
| `rolling_sharpe(returns)` | 滚动夏普比率 |
| `rolling_volatility(returns)` | 滚动波动率 |

### 报告生成 (reports.py)

| 函数 | 说明 |
|------|------|
| `html(returns, benchmark, output)` | 生成 HTML tearsheet 报告 |
| `full(returns, benchmark)` | 全面绩效分析报告（终端/Notebook） |
| `basic(returns, benchmark)` | 基础绩效报告 |
| `metrics(returns, benchmark, mode)` | 绩效指标表格 |

## 版本与兼容性

| 组件 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| pandas | 1.5+ (已兼容 2.2.0+ 的频率别名变更) |
| numpy | 1.25+ |
| matplotlib | 3.5+ |
| seaborn | 0.12+ |

---

## 许可证

本项目继承自 [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats)，使用 **Apache License 2.0**。

---

## 主要改动记录

- 所有 docstring 和代码注释已翻译为中文
- 报告输出支持中文本地化（指标名称、单位等）
- 移除了在线数据下载依赖（yfinance 等），仅支持本地数据输入
- 移除了当前项目未使用的蒙特卡洛模拟支线，使报告增强库聚焦绩效指标与图表
- 兼容 pandas 2.2.0+ 的频率别名变更
- 兼容 NumPy 1.25+ 的已弃用函数
