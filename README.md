# FactorForge period-offset 分支

这个 worktree 对应 `version/period-offset` 分支。它基于 `main` 的因子分析与模拟回测框架，增加了周期换仓日预计算、offset 管理和多 offset ensemble，适合研究换仓起点对策略结果的影响。

## 与 main 分支的关系

- `main`：稳定基线分支，使用传统 `turnover` 口径，适合常规日频/多日持有期回测和报告输出。
- `version/period-offset`：实验扩展分支，使用 `period` 和 `offsets` 替代单一换仓日逻辑，支持同周期多 offset 同时运行。

本分支也同步了主分支的报告优化：中文图例、中文标题、中文指标表，benchmark 使用黑色实线，Step1 统计表删除重复 ICIR 列。

## period 预运算

本分支使用 `generate_period_files.py` 生成换仓日矩阵。生成结果由配置中的 `paths.period_path` 指向，通常包含：

- `period.npy`：周期键名，例如 `5_0`、`5_1`、`20_7`、`W_0`；
- `period_dates.npy`：每个周期键对应的换仓日布尔矩阵；
- `dates.npy`：与换仓日矩阵对齐的交易日序列。

Step1 和 Step2 都通过 `period_path` 读取这些预运算结果，避免在每个因子、每个 offset 上重复推导换仓日。

## 配置字段

`analysis` 和 `simulation` 中的关键字段如下：

- `period`：换仓周期，可写 `1`、`5`、`20`、`W`、`2W`、`M` 等。
- `offsets`：换仓起点。写 `[0]` 表示只跑单个 offset；写 `all` 表示使用该 period 下全部 offset。
- `offset_mode`：目前使用 `ensemble`，表示多 offset 分资金袖套合成。
- `label_days`：Step1 的 IC/RankIC 标签周期，可写 `1`、`5`、`20`、`W` 等，对应 `vwap_1d.npy`、`vwap_5d.npy`、`vwap_W.npy`。

示例：

```yaml
analysis:
  period: 5
  offsets: all
  offset_mode: ensemble
  label_days: 5

simulation:
  period: 5
  offsets: all
  offset_mode: ensemble
```

这会使用 `5_0` 到 `5_4` 的预计算换仓日，并在 Step2 中按 offset 分袖套交易后合成总净值。

## 常用命令

```powershell
python generate_period_files.py --output D:/凯纳/原始数据/period
python step1_因子分析.py --config config/kaiyuan.yaml
python step2_模拟回测.py --config config/kaiyuan.yaml
```

`config/config.yaml` 和 `config/kaiyuan.yaml` 是保留的示例配置。其他批量实验 YAML 默认被 `.gitignore` 忽略。

## 输出报告

Step1 输出 IC、RankIC、滚动 t 值、分层净值、换手率和统计表。Step2 输出策略净值、基准净值、超额净值、回撤、日收益率、日换手率、选股画像，以及本分支特有的 offset 净值对比图。

当 `offsets: all` 时，建议同时查看：

- ensemble 后的主资金曲线；
- 各 offset 的净值对比；
- 不同 offset 的回撤和换手差异。
