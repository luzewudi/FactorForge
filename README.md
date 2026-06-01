# FactorForge

FactorForge 是一个 YAML 驱动的因子研究与模拟回测框架。当前仓库保留两条主要分支：

- `main`：稳定基线分支，使用固定日数换仓参数 `turnover`，适合常规日频因子分析、分层净值检验和单账户模拟回测。
- `version/period-offset`：周期与 offset 扩展分支，引入预计算换仓日、`period`、`offsets` 和多 offset ensemble，适合对周频、月频或不同起始换仓日进行稳健性比较。

## main 分支

`main` 分支是默认生产口径，代码路径更直接：

1. `step1_因子分析.py` 读取 `config/*.yaml` 中的 `analysis` 配置，完成 IC、RankIC、滚动 t 值、分层净值、分组换手和 HTML 报告。
2. `step2_模拟回测.py` 读取同一份配置中的 `simulation` 配置，基于 Step1 的 TOTAL RankIC 均值自动判断选股方向，输出资金曲线、交易明细、选股结果、选股画像和 HTML 报告。
3. 标签文件按 `trade_price + turnover` 自动匹配，例如 `vwap_1d.npy`、`vwap_5d.npy`。

推荐在这个分支上做稳定复现、报告展示、样式修正和常规因子批量回测。

## version/period-offset 分支

`version/period-offset` 分支在主流程上增加了周期调度能力。核心差异是：

1. 增加 `generate_period_files.py`，用于生成 `period.npy`、`period_dates.npy`、`dates.npy` 等换仓日预运算文件。
2. 配置从 `turnover` 扩展为 `period`、`offsets`、`offset_mode`、`label_days`。
3. `offsets: all` 可以在同一个 period 下跑全部 offset，并把不同 offset 的资金袖套合成为 ensemble。
4. Step2 报告额外支持 offset 净值对比图，便于观察调仓起点敏感性。

这个分支适合研究“同一个因子在不同换仓起点下是否稳定”，例如：

- `period: 5, offsets: all` 使用 `5_0` 到 `5_4`；
- `period: 20, offsets: all` 使用 `20_0` 到 `20_19`；
- `period: W` 或 `period: M` 用于周频、月频类换仓。

## 常用命令

```powershell
python step1_因子分析.py --config config/kaiyuan.yaml
python step2_模拟回测.py --config config/kaiyuan.yaml
```

`config/config.yaml` 和 `config/kaiyuan.yaml` 是保留的示例配置。其他临时或批量运行配置默认被 `.gitignore` 忽略，避免把本地实验参数误提交。

## 报告说明

HTML 报告默认使用中文标题和中文指标表。Step2 中 benchmark 使用黑色实线，策略、基准、超额净值和回撤在同一张图里展示；日收益率、日换手率和选股画像会作为独立子图输出。

Step1 的统计表保留通用列 `ICIR/RankICIR` 与 `年化 ICIR/RankICIR`，不再额外重复输出 `ICIR`、`RankICIR`、`Annualized_ICIR`、`Annualized_RankICIR` 四列。
