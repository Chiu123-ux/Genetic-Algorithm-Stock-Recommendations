#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动取最近交易日，更新中证500数据并执行预测（可选重训）"
    )
    parser.add_argument("--start", default="2019-01-01", help="数据起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--datahub-path", default="/home/intership/public/datahub/", help="datahub 路径")
    parser.add_argument("--cpu-num", type=int, default=100, help="构建数据并发参数")
    parser.add_argument("--probe-days", type=int, default=14, help="探测最近交易日的回看天数")
    parser.add_argument(
        "--config",
        default="config_predict_zz500.json",
        help="预测配置文件（相对项目根目录）",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="是否在推荐前重跑 train-models（默认不重训，适合只更新1天数据）",
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="项目根目录",
    )
    return parser.parse_args()


def detect_latest_trade_date(datahub_path: str, probe_days: int, cpu_num: int) -> str:
    if datahub_path not in sys.path:
        sys.path.append(datahub_path)

    read_datahub = importlib.import_module("read_datahub")
    datahub_daily = getattr(read_datahub, "datahub_daily")

    today = datetime.now().date()
    start = (today - timedelta(days=probe_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    probe_fields = ["ZZ500_data", "close_adj", "close"]
    latest_date = None

    for field in probe_fields:
        try:
            data = datahub_daily([field], start, end, cpu_num=cpu_num)
            frame = data.get(field)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                idx = pd.to_datetime(frame.index, errors="coerce")
                idx = idx[~idx.isna()]
                if len(idx) > 0:
                    candidate = idx.max().date()
                    # 检查 000905-SH-index.weight.csv 是否存在
                    weight_file = Path("/data/hft/Daily/Wind/index") / candidate.strftime("%Y/%m/%d") / "000905-SH-index.weight.csv"
                    if not weight_file.exists():
                        print(f"[WARN] 探测到日期 {candidate}，但成分股文件不存在: {weight_file}")
                        continue
                    
                    if latest_date is None or candidate > latest_date:
                        latest_date = candidate
        except Exception:
            continue

    if latest_date is None:
        return end
    return latest_date.strftime("%Y-%m-%d")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_recommendation_dir(project_dir: Path, config_path: Path) -> str | None:
    try:
        cfg_file = config_path if config_path.is_absolute() else (project_dir / config_path)
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        return cfg.get("gru_pipeline", {}).get("recommendation_output_dir")
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    build_script = project_dir / "scripts" / "build_pickles_from_datahub.py"
    run_pipeline = project_dir / "run_pipeline.sh"
    config_path = Path(args.config)

    latest_trade_date = detect_latest_trade_date(args.datahub_path, args.probe_days, args.cpu_num)
    print(f"[INFO] 最近交易日: {latest_trade_date}")

    close_path = f"/home/intership/guojh/GA和GRU/stock_factors_predict_{latest_trade_date}_close_zz500.pkl"
    factor_path = f"/home/intership/guojh/GA和GRU/stock_factors_predict_{latest_trade_date}_factors_zz500.pkl"

    cfg_file = config_path if config_path.is_absolute() else (project_dir / config_path)
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "ga_pipeline" in cfg:
            cfg["ga_pipeline"]["price_file"] = close_path
            cfg["ga_pipeline"]["factor_file"] = factor_path
        if "gru_pipeline" in cfg:
            cfg["gru_pipeline"]["price_file"] = close_path
            cfg["gru_pipeline"]["factor_file"] = factor_path
            cfg["gru_pipeline"]["backtest_price_file"] = close_path
            cfg["gru_pipeline"]["backtest_factor_file"] = factor_path
            # 设置推荐结果输出目录
            cfg["gru_pipeline"]["recommendation_output_dir"] = "/home/intership/guojh/GA和GRU/炒股"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 已更新配置文件 {cfg_file.name} 中的数据路径和推荐输出目录")
    except Exception as e:
        print(f"[WARN] 无法更新配置文件: {e}")

    run_cmd(
        [
            sys.executable,
            str(build_script),
            "--start",
            args.start,
            "--end",
            latest_trade_date,
            "--cpu-num",
            str(args.cpu_num),
            "--datahub-path",
            args.datahub_path,
            "--universe",
            "zz500",
            "--output-close",
            close_path,
            "--output-factors",
            factor_path,
            "--metadata-job",
            "zz500_predict_latest",
        ],
        project_dir,
    )

    if args.retrain:
        run_cmd(["bash", str(run_pipeline), "--config", args.config, "train-models"], project_dir)
    else:
        print("[INFO] 跳过重训（使用已有模型）")

    run_cmd(["bash", str(run_pipeline), "--config", args.config, "gru-recommend"], project_dir)

    out_dir = load_recommendation_dir(project_dir, config_path)
    if out_dir:
        print(f"[DONE] 推荐结果目录: {out_dir}")
    else:
        print("[DONE] 推荐已完成，请查看配置中的 gru_pipeline.recommendation_output_dir")


if __name__ == "__main__":
    main()
