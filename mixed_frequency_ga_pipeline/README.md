# Mixed-Frequency GA + GRU Pipeline

这是一个可上传 GitHub 的精简代码包，覆盖：

```text
数据准备
  -> 日频 + 分钟频率混频数据
  -> 3D Native 遗传规划（GA）挖掘因子
  -> 生成因子表达式
  -> GRU 训练
  -> GA + GRU 回测
  -> 最新预测与推荐
```

本目录只包含代码、配置模板和因子表达式模板，不包含原始行情数据、分钟缓存、模型权重、历史日志和实验报告。

## 目录

```text
configs/       脱敏后的 GA、回测和预测配置模板
src/           主流程和核心管线
gplearn/       项目内置遗传规划实现
toolkit/       GA 依赖的函数、数据和回测工具
scripts/       数据准备、因子评估、回测和预测辅助脚本
factors/       因子表达式模板；运行结果也建议放在这里
data/          本地数据挂载点，不提交真实数据
models/        模型输出挂载点，不提交模型权重
outputs/       运行输出挂载点
reports/       精简后的结果输出挂载点
```

## 安装

建议使用 Python 3.11，并按照目标机器安装匹配 CPU 或 CUDA 版本的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch 是 GRU 必需依赖，版本应根据目标机器和 CUDA 版本从官方渠道安装；`requirements.txt` 不锁定 CUDA 专用 wheel。例如 CPU 环境可单独执行：

```bash
pip install torch
```

GPU 环境请按照 PyTorch 官方安装命令选择对应 CUDA wheel。

## 数据准备

配置模板中的路径已经改为相对路径，但数据仍需由使用者准备。至少需要：

- 混频价格文件，例如 `data/mixed_close.pkl`
- 混频因子文件，例如 `data/mixed_factors.pkl`
- 分钟缓存目录，例如 `data/minute_cache/`
- 外部目标数据目录，例如 `data/targets/`
- 股票池和其他交易日辅助文件（如果配置启用）

不要把真实行情、分钟缓存、模型权重或含有内部路径的数据文件上传到公开 GitHub 仓库。建议在 `data/` 中只提交字段说明或 manifest。

## 主流程

所有命令都在仓库根目录执行。统一入口是：

```bash
python -m src.main --config configs/config_mixed_ga.json mine-ga
python -m src.main --config configs/config_mixed_ga.json train-gru
python -m src.main --config configs/config_backtest.json ga-gru-backtest
python -m src.main --config configs/config_predict.json gru-recommend
```

也可以使用：

```powershell
.\run_pipeline.ps1 --config configs\config_mixed_ga.json mine-ga
```

命令含义：

- `mine-ga`：读取混频数据并挖掘 GA 因子，输出 `generated_factors_latest.txt`。
- `train-gru`：读取因子表达式和数据，训练 GRU。
- `ga-gru-backtest`：执行滚动训练/回测并写出结果。
- `gru-recommend`：使用最新数据生成股票推荐。

如果数据尚未转换为本项目使用的格式，可先参考：

```bash
python scripts/build_pickles_from_datahub.py --help
python scripts/build_daily_factors_from_intraday.py --help
python -m src.main --config configs/config_mixed_ga.json build-minute-cache --help
```

这些数据准备步骤依赖外部 DataHub 或行情接口，不属于本代码包。

## 配置文件

- `configs/config_mixed_ga.json`：3D Native 混频 GA 和 GRU 的主配置模板。
- `configs/config_backtest.json`：ZZ500 回测配置模板。
- `configs/config_predict.json`：ZZ500 预测配置模板。

上传前请检查并修改：

- `price_file`
- `factor_file`
- `minute_cache_root`
- `target_parquet_root`
- `factor_expression_file`
- `output_directory`
- `model_output_dir`
- `recommendation_output_dir`

配置中的 `./data`、`./outputs`、`./models` 等路径是模板路径，不代表仓库内已经包含数据。

## 输出约定

建议把运行结果放在以下位置：

```text
factors/generated_factors_latest.txt
models/
outputs/ga/
outputs/backtest/
outputs/predict/
reports/
```

这些目录默认被 `.gitignore` 忽略。正式研究结果应单独整理成脱敏摘要后再提交。

## GitHub 发布前检查

```bash
git status --short
git grep -n -E '(/home/|/data/|/raid0/|[A-Za-z]:\\|token|password|secret|api_key)'
```

确认以下内容没有被提交：

- `.venv/`、`.conda*/`
- `__pycache__/`、`*.pyc`
- 原始行情和分钟缓存
- 模型权重
- 运行日志和大批量 CSV/PNG
- 内部服务器路径、账号、Token 或 API 密钥

## 说明

本包没有附带 smoke test 或历史实验结果；它是面向 GitHub 发布的精简代码包。运行前需要在目标机器准备数据并修改配置。
