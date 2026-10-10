"""Command line entry points for the modernised pipelines."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from src.data_fetch.close_price_fetcher import run_close_price_fetch
except Exception:  # pragma: no cover - optional dependency
    run_close_price_fetch = None

try:
    from src.pipelines.factor_pipeline import run_factor_job
except Exception:  # pragma: no cover - optional dependency
    run_factor_job = None
from src.pipelines.ga_factor_miner import GAFactorMiner, load_ga_config
from src.pipelines.gru_forecaster import GRUFactorForecaster, load_gru_config
from src.pipelines.gru_backtester import GRUBacktester, load_gru_backtest_config

try:
    from src.pipelines.smart_volume_predictor import SmartVolumeEnhancedPredictor, load_smart_volume_config
    from src.pipelines.smart_volume_backtester import SmartVolumeBacktester, load_backtest_config
except Exception:  # pragma: no cover - optional dependency
    SmartVolumeEnhancedPredictor = None
    load_smart_volume_config = None
    SmartVolumeBacktester = None
    load_backtest_config = None
from src.utils.data_io import load_price_and_factors, load_price_factor_bundle
from src.utils.minute_tensor_cache import DEFAULT_CACHE_FORMAT, DEFAULT_SOURCE_ROOT, build_minute_cache_range
from src.utils.config_loader import set_config_override


def _load_stock_whitelist_file(path: str | None) -> List[str] | None:
    if not path:
        return None
    whitelist_path = Path(path)
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Stock whitelist file not found: {whitelist_path}")

    values = [line.strip() for line in whitelist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return values or None


def fetch_factors(job: str, output: str | None) -> None:
    if run_factor_job is None:
        raise RuntimeError("factor fetch modules are unavailable in this environment")
    summary = run_factor_job(job, output)
    _print_factor_summary(summary)


def _print_factor_summary(summary) -> None:
    print(f"Job: {summary.job}")
    print(f"Factor file: {summary.factor_file}")
    if summary.close_file:
        print(f"Close file: {summary.close_file}")
    print(f"Latest factor file: {summary.latest_factor_file}")
    if summary.latest_close_file:
        print(f"Latest close file: {summary.latest_close_file}")
    if summary.chunk_files:
        print("Chunks:")
        for path in summary.chunk_files:
            print(f"  - {path}")
    print(f"Stock count: {summary.stock_count}")
    print(f"Date range: {summary.date_range[0]} -> {summary.date_range[1]}")


def _require_non_empty(name: str, value: str | None) -> None:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Target preflight failed: '{name}' is empty")


def _print_target_snapshot(command: str) -> None:
    """Print and validate target-related config to avoid running stale/legacy settings by mistake."""
    ga_cfg = load_ga_config()
    gru_cfg = load_gru_config()
    backtest_cfg = load_gru_backtest_config()

    ga_mode = str(getattr(ga_cfg, "target_mode", "legacy_price_return")).lower()
    gru_mode = str(getattr(gru_cfg, "target_mode", "legacy_price_return")).lower()
    bt_mode = str(getattr(backtest_cfg, "backtest_return_mode", "daily_nextday")).lower()

    print("[Preflight] Target config snapshot")
    print(
        "  GA  -> "
        f"mode={ga_mode}, root={ga_cfg.target_parquet_root}, "
        f"train_field={ga_cfg.target_train_field}, signal_to_entry_days={ga_cfg.signal_to_entry_days}"
    )
    print(
        "  GRU -> "
        f"mode={gru_mode}, root={gru_cfg.target_parquet_root}, "
        f"train_field={gru_cfg.target_train_field}, "
        f"backtest_root={gru_cfg.target_backtest_parquet_root or gru_cfg.target_parquet_root}, "
        f"backtest_field={gru_cfg.target_backtest_field}, "
        f"signal_to_entry_days={gru_cfg.signal_to_entry_days}"
    )
    print(f"  BT  -> backtest_return_mode={bt_mode}")

    ga_external = ga_mode != "legacy_price_return"
    gru_external = gru_mode != "legacy_price_return"

    if ga_external:
        _require_non_empty("ga_pipeline.target_parquet_root", ga_cfg.target_parquet_root)
        _require_non_empty("ga_pipeline.target_train_field", ga_cfg.target_train_field)
        if not Path(str(ga_cfg.target_parquet_root)).exists():
            raise ValueError(
                f"Target preflight failed: ga_pipeline.target_parquet_root does not exist: {ga_cfg.target_parquet_root}"
            )

    if gru_external:
        _require_non_empty("gru_pipeline.target_parquet_root", gru_cfg.target_parquet_root)
        _require_non_empty("gru_pipeline.target_train_field", gru_cfg.target_train_field)
        _require_non_empty("gru_pipeline.target_backtest_field", gru_cfg.target_backtest_field)
        if not Path(str(gru_cfg.target_parquet_root)).exists():
            raise ValueError(
                f"Target preflight failed: gru_pipeline.target_parquet_root does not exist: {gru_cfg.target_parquet_root}"
            )
        backtest_root = gru_cfg.target_backtest_parquet_root or gru_cfg.target_parquet_root
        if backtest_root and not Path(str(backtest_root)).exists():
            raise ValueError(
                "Target preflight failed: gru_pipeline.target_backtest_parquet_root does not exist: "
                f"{backtest_root}"
            )

    # For GA/GRU/backtest workflows, require GA & GRU to stay on the same target family.
    if command in {"train-models", "ga-gru-backtest"}:
        if ga_mode != gru_mode:
            raise ValueError(
                "Target preflight failed: ga_pipeline.target_mode and gru_pipeline.target_mode mismatch"
            )
        if ga_external and gru_external and str(ga_cfg.target_parquet_root) != str(gru_cfg.target_parquet_root):
            raise ValueError(
                "Target preflight failed: ga/gru target_parquet_root mismatch"
            )
        if int(ga_cfg.signal_to_entry_days) != int(gru_cfg.signal_to_entry_days):
            raise ValueError(
                "Target preflight failed: ga/gru signal_to_entry_days mismatch"
            )

    if command == "ga-gru-backtest":
        if bt_mode in {"rebalance_5d", "daily_external_target"} and not gru_external:
            raise ValueError(
                "Target preflight failed: external backtest_return_mode requires gru_pipeline.target_mode != legacy_price_return"
            )
        if bt_mode not in {"daily_nextday", "rebalance_5d", "daily_external_target"}:
            raise ValueError(
                "Target preflight failed: unsupported gru_pipeline.backtest_return_mode"
            )


def run_ga_miner() -> None:
    config = load_ga_config()
    miner = GAFactorMiner(config)
    print(
        "GA target config -> "
        f"mode={config.target_mode}, field={config.target_train_field}, "
        f"signal_to_entry_days={config.signal_to_entry_days}, horizon={config.target_horizon_days}"
    )
    expressions = miner.mine()
    print("Generated factors:")
    for item in expressions:
        print(f"  IC={item.score:.4f} :: {item.expression}")
    latest_file = Path(config.output_directory) / "generated_factors_latest.txt"
    print(f"Latest factors stored at {latest_file}")


def run_smart_volume_live() -> Tuple[pd.Series, Path, int]:
    if SmartVolumeEnhancedPredictor is None or load_smart_volume_config is None:
        raise RuntimeError("smart volume modules are unavailable in this environment")
    config = load_smart_volume_config()
    predictor = SmartVolumeEnhancedPredictor(config)
    price_df, factors = load_price_and_factors(config.price_file, config.factor_file)
    predictor.train(price_df, factors)
    scores = predictor.predict(price_df, factors)
    report_path = predictor.save_recommendations(scores)
    return scores, report_path, config.recommendation_count


def run_smart_volume_backtest() -> dict[str, Path]:
    if (
        SmartVolumeEnhancedPredictor is None
        or SmartVolumeBacktester is None
        or load_smart_volume_config is None
        or load_backtest_config is None
    ):
        raise RuntimeError("smart volume modules are unavailable in this environment")
    sv_config = load_smart_volume_config()
    backtest_config = load_backtest_config()
    predictor = SmartVolumeEnhancedPredictor(sv_config)
    backtester = SmartVolumeBacktester(backtest_config)

    price_df, factors = load_price_and_factors(sv_config.price_file, sv_config.factor_file)
    result = backtester.run(price_df, factors, predictor)
    report_dir = Path(sv_config.report_directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"smart_volume_backtest_{timestamp}.csv"
    result.to_csv(report_path, index=False)
    performance = {
        "curve_path": report_path,
    }
    if not result.empty:
        values = result["value"].astype(float)
        dates = pd.to_datetime(result["date"]) if "date" in result else pd.Series(dtype="datetime64[ns]")
        returns = values.pct_change().fillna(0.0)
        total_return = float(values.iloc[-1] / values.iloc[0] - 1)
        period_days = float((dates.iloc[-1] - dates.iloc[0]).days) if not dates.empty else 0.0
        years = period_days / 365.0 if period_days > 0 else 0.0
        annualized_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else total_return
        volatility = float(returns.std() * np.sqrt(252))
        cum_max = values.cummax()
        drawdown = (values / cum_max) - 1
        max_drawdown = float(drawdown.min())
        summary = {
            "start_date": str(result["date"].iloc[0]) if "date" in result else "",
            "end_date": str(result["date"].iloc[-1]) if "date" in result else "",
            "period_days": period_days,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
        }
        summary_path = report_dir / f"smart_volume_performance_{timestamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        performance["summary_path"] = summary_path

        try:
            import matplotlib.pyplot as plt

            plt.switch_backend("Agg")
            norm_values = values / values.iloc[0]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, norm_values, label="Strategy")
            ax.set_title("Smart Volume Backtest Cumulative Return")
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalised Portfolio Value")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            plot_path = report_dir / f"smart_volume_performance_{timestamp}.png"
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            performance["plot_path"] = plot_path
        except Exception as exc:  # pragma: no cover - optional plotting
            print(f"Warning: failed to export backtest plot ({exc})")

    print(result.tail())
    print(f"Backtest curve saved to {report_path}")
    if "summary_path" in performance:
        print(f"Performance summary saved to {performance['summary_path']}")
    if "plot_path" in performance:
        print(f"Performance chart saved to {performance['plot_path']}")
    return performance


def train_gru_forecaster() -> None:
    config = load_gru_config()
    if not config.factor_expression_file:
        raise ValueError("GRU pipeline requires GA factor expressions; run mine-ga or train-models first.")
    forecaster = GRUFactorForecaster(config)
    print(
        "GRU target config -> "
        f"mode={config.target_mode}, train_field={config.target_train_field}, "
        f"signal_to_entry_days={config.signal_to_entry_days}, horizon={config.target_horizon_days}"
    )
    price_df, factors = forecaster.load_data()
    expressions = forecaster.load_factor_expressions()
    X, y, target_dates = forecaster.create_dataset(price_df, factors, expressions)
    outputs = forecaster.train(X, y, sample_dates=target_dates)
    print("GRU model trained successfully.")
    print(f"Model saved to {outputs['model_path']}")
    print(f"Latest checkpoint: {outputs['latest_path']}")
    print(f"Training log: {outputs['log_path']}")


def run_gru_recommendation() -> None:
    config = load_gru_config()
    if not config.factor_expression_file:
        raise ValueError("GRU pipeline requires GA factor expressions; run mine-ga or train-models first.")
    forecaster = GRUFactorForecaster(config)
    price_df, factors = forecaster.load_data()
    expressions = forecaster.load_factor_expressions()
    ranked, path = forecaster.recommend(price_df, factors, expressions)
    top_display = ranked.head(config.top_n or 10)
    print("Latest GRU recommendations (descending score):")
    print(top_display.to_string())
    if path is not None:
        print(f"Recommendation file saved to {path}")


def _resolve_initial_ga_cutoff_date(price_index: pd.Index, initial_train_months: int) -> pd.Timestamp:
    dates = pd.to_datetime(price_index)
    if dates.empty:
        raise ValueError("Cannot initialise GA for backtest: backtest price data is empty")

    months = max(0, int(initial_train_months or 0))
    target_date = pd.Timestamp(dates[0]) + pd.DateOffset(months=months)
    cutoff_pos = int(dates.searchsorted(target_date, side="left"))
    if cutoff_pos >= len(dates):
        raise ValueError(
            "Cannot initialise GA for backtest: initial_train_months exceeds available backtest history"
        )
    return pd.Timestamp(dates[cutoff_pos])


def run_ga_gru_backtest() -> Dict[str, Path]:
    config = load_gru_config()
    model_type = str(getattr(config, "model_type", "gru") or "gru").strip().lower()
    model_label = model_type.upper()
    backtest_config = load_gru_backtest_config()
    if not config.factor_expression_file:
        raise ValueError("GA factor expressions missing; run GA mining before the GA+GRU backtest.")

    price_source = config.backtest_price_file or config.price_file
    factor_source = config.backtest_factor_file or config.factor_file
    # 回测时统一使用 backtest 数据源，避免后续滚动训练/重载时与 X/y 的股票维度不一致
    config.price_file = price_source
    config.factor_file = factor_source

    forecaster = GRUFactorForecaster(config)
    price_df, factors = forecaster.load_data(price_source, factor_source)
    _, _, metadata, benchmarks = load_price_factor_bundle(price_source, factor_source)

    initial_ga_cutoff_date = _resolve_initial_ga_cutoff_date(
        price_df.index,
        int(getattr(backtest_config, "initial_train_months", 0) or 0),
    )
    print(
        "Bootstrapping GA factor expressions -> "
        f"initial_train_months={backtest_config.initial_train_months}, cutoff_date={initial_ga_cutoff_date.date()}"
    )
    ga_cfg = load_ga_config()
    ga_cfg.predict_days = config.predict_days
    miner = GAFactorMiner(ga_cfg)
    miner.mine(
        pop_num=backtest_config.backtest_ga_pop_num,
        gen_num=backtest_config.backtest_ga_gen_num,
        cutoff_date=initial_ga_cutoff_date,
    )

    setattr(backtest_config, "initial_ga_refresh_date", initial_ga_cutoff_date)

    expressions = forecaster.load_factor_expressions()
    X, y, dates = forecaster.create_dataset(price_df, factors, expressions)

    universe_name = str(metadata.get("universe", "")).lower()
    benchmark_name = None
    if universe_name in benchmarks:
        benchmark_name = universe_name
    elif universe_name == "zz500_zz1000" and "zz500" in benchmarks:
        benchmark_name = "zz500"
    elif "zz500" in benchmarks:
        benchmark_name = "zz500"
    elif "hs300" in benchmarks:
        benchmark_name = "hs300"

    backtester = GRUBacktester(
        config,
        backtest_config,
        list(price_df.columns),
        benchmark_series=benchmarks,
        benchmark_name=benchmark_name,
    )
    print(
        f"GA+{model_label} backtest -> rolling train window: {backtest_config.train_window_months} months "
        f"(min {backtest_config.min_train_samples} samples), "
        f"retrain every {backtest_config.retrain_every_n_rebalances} rebalances, "
        f"top {backtest_config.top_n} selections, transaction cost {backtest_config.transaction_cost:.4f}, "
        f"target_mode={config.target_mode}, backtest_return_mode={backtest_config.backtest_return_mode}"
    )
    result, summary = backtester.run(X, y, dates)

    report_dir = Path(config.log_output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    curve_path = report_dir / f"ga_gru_backtest_{timestamp}.csv"
    result.to_csv(curve_path, index=False)

    summary_path = report_dir / f"ga_gru_performance_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    outputs: Dict[str, Path] = {"curve_path": curve_path, "summary_path": summary_path}

    print(result.tail())
    try:
        total_return_pct = f"{summary['total_return'] * 100:.2f}%"
        annualized_pct = f"{summary['annualized_return'] * 100:.2f}%"
        max_dd_pct = f"{summary['max_drawdown'] * 100:.2f}%"
        benchmark_text = ""
        if summary.get("benchmark_name") and summary.get("benchmark_total_return") is not None:
            benchmark_total_pct = f"{summary['benchmark_total_return'] * 100:.2f}%"
            excess_total_pct = f"{summary.get('excess_total_return', 0.0) * 100:.2f}%"
            excess_spread_pct = f"{summary.get('excess_return_spread', 0.0) * 100:.2f}%"
            benchmark_text = (
                f", Benchmark({summary['benchmark_name']}): {benchmark_total_pct}, "
                f"Excess: {excess_total_pct}, Spread: {excess_spread_pct}"
            )
        print(
            f"GA+{model_label} headline metrics -> "
            f"Total Return: {total_return_pct}, Annualised: {annualized_pct}, Max Drawdown: {max_dd_pct}"
            f"{benchmark_text}"
        )
    except Exception:  # pragma: no cover - best effort formatting
        pass
    print(f"GA+{model_label} backtest curve saved to {curve_path}")
    print(f"GA+{model_label} performance summary saved to {summary_path}")

    try:
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        dates_series = pd.to_datetime(result["date"])
        values = result["portfolio_value"].astype(float)
        norm_values = values / summary["initial_capital"]
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.6]},
        )
        axes[0].plot(dates_series, norm_values, label="GA+GRU Strategy", linewidth=1.8)

        if "benchmark_curve" in result and result["benchmark_curve"].notna().any():
            axes[0].plot(
                dates_series,
                result["benchmark_curve"].astype(float),
                label=f"Benchmark ({summary.get('benchmark_name', 'primary')})",
                linewidth=1.4,
            )

        for benchmark_name in sorted(benchmarks):
            curve_col = f"benchmark_curve_{benchmark_name}"
            if curve_col not in result:
                continue
            if benchmark_name == summary.get("benchmark_name"):
                continue
            axes[0].plot(
                dates_series,
                result[curve_col].astype(float),
                label=f"Benchmark ({benchmark_name})",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )

        axes[0].set_title("GA+GRU Backtest vs Benchmarks")
        axes[0].set_ylabel("Normalised Value")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].legend()

        if "excess_curve" in result and result["excess_curve"].notna().any():
            axes[1].plot(
                dates_series,
                result["excess_curve"].astype(float),
                label=f"Excess vs {summary.get('benchmark_name', 'benchmark')}",
                color="tab:orange",
                linewidth=1.6,
            )
            axes[1].axhline(1.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
            axes[1].legend()

        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Excess Curve")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        plot_path = report_dir / f"ga_gru_performance_{timestamp}.png"
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        outputs["plot_path"] = plot_path
        print(f"GA+GRU performance chart saved to {plot_path}")
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Warning: failed to export GA+GRU backtest plot ({exc})")

    return outputs


def run_data_preparation_flow(jobs: List[str]) -> None:
    if run_factor_job is None:
        raise RuntimeError("factor fetch modules are unavailable in this environment")
    print("Starting data preparation flow...")
    # close_path = run_close_price_fetch()
    # print(f"Close price data saved to {close_path}")
    for job in jobs:
        print(f"\nRunning factor fetch job: {job}")
        summary = run_factor_job(job, None)
        _print_factor_summary(summary)


def run_model_training_flow() -> None:
    print("Starting model training flow...")
    run_ga_miner()
    train_gru_forecaster()


def run_performance_test_flow() -> None:
    print("Starting performance testing flow...")
    run_smart_volume_backtest()


def run_live_demo_flow() -> None:
    print("Running live recommendation demo...")
    scores, report_path, top_n = run_smart_volume_live()
    print(f"Recommendations saved to {report_path}")
    top_preview = scores.head(top_n)
    print("Top candidates:")
    print(top_preview.to_string())


def run_build_minute_cache(args: argparse.Namespace) -> None:
    fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    stock_whitelist = _load_stock_whitelist_file(getattr(args, "stock_whitelist_file", None))
    summary = build_minute_cache_range(
        source_root=args.source_root,
        cache_root=args.output_root,
        start_date=args.start,
        end_date=args.end,
        fields=fields,
        start_time=args.start_time,
        end_time=args.end_time,
        stock_whitelist=stock_whitelist,
        dtype=args.dtype,
        cache_format=args.cache_format,
        overwrite=args.overwrite,
    )
    print("Minute tensor cache build completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="gplearn modernised pipelines")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to alternative config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-factors", help="Fetch factor datasets via TuShare")
    fetch_parser.add_argument("job", help="Job name as configured in config.json")
    fetch_parser.add_argument("--output", help="Override output directory", default=None)

    subparsers.add_parser("mine-ga", help="Run GA based factor mining")
    subparsers.add_parser("smart-volume-live", help="Generate live smart volume recommendations")
    subparsers.add_parser("smart-volume-backtest", help="Backtest smart volume strategy")
    subparsers.add_parser("train-gru", help="Train GRU forecaster using GA factors")
    subparsers.add_parser("gru-recommend", help="Generate GRU-based stock recommendations")
    subparsers.add_parser("ga-gru-backtest", help="Run GA+GRU factor strategy walk-forward backtest")
    subparsers.add_parser("fetch-close", help="Fetch closing price history")
    minute_cache_parser = subparsers.add_parser("build-minute-cache", help="Build per-day minute tensor cache files")
    minute_cache_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    minute_cache_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    minute_cache_parser.add_argument("--fields", required=True, help="Comma separated minute field names")
    minute_cache_parser.add_argument("--output-root", required=True, help="Output cache root directory")
    minute_cache_parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Minute HDF source root")
    minute_cache_parser.add_argument("--start-time", default=None, help="Optional start minute, e.g. 0930")
    minute_cache_parser.add_argument("--end-time", default=None, help="Optional end minute, e.g. 1500")
    minute_cache_parser.add_argument("--stock-whitelist-file", default=None, help="Optional text file containing one stock code per line; only these stocks will be cached")
    minute_cache_parser.add_argument("--dtype", default="float32", help="Stored numeric dtype")
    minute_cache_parser.add_argument("--cache-format", default=DEFAULT_CACHE_FORMAT, choices=["npy", "npz"], help="Cache storage format; npy is mmap-friendly and recommended for long windows")
    minute_cache_parser.add_argument("--overwrite", action="store_true", help="Rebuild cache entries even if they exist")
    prep_parser = subparsers.add_parser("prepare-data", help="Run the full data preparation flow")
    prep_parser.add_argument("--jobs", nargs="+", default=["live"], help="Factor fetch jobs to execute")
    subparsers.add_parser("train-models", help="Run GA and GRU training pipeline")
    subparsers.add_parser("performance-test", help="Run smart volume backtest and export reports")
    subparsers.add_parser("live-demo", help="Run live recommendation demo using trained models")

    args = parser.parse_args()

    set_config_override(args.config_path)

    if args.command in {"mine-ga", "train-gru", "ga-gru-backtest", "train-models"}:
        _print_target_snapshot(args.command)

    if args.command == "fetch-factors":
        fetch_factors(args.job, args.output)
    elif args.command == "mine-ga":
        run_ga_miner()
    elif args.command == "smart-volume-live":
        scores, report_path, top_n = run_smart_volume_live()
        print(f"Recommendations saved to {report_path}")
        print("Top candidates:")
        print(scores.head(top_n).to_string())
    elif args.command == "smart-volume-backtest":
        run_smart_volume_backtest()
    elif args.command == "train-gru":
        train_gru_forecaster()
    elif args.command == "gru-recommend":
        run_gru_recommendation()
    elif args.command == "ga-gru-backtest":
        run_ga_gru_backtest()
    elif args.command == "fetch-close":
        if run_close_price_fetch is None:
            raise RuntimeError("close price fetch module is unavailable in this environment")
        output_path = run_close_price_fetch()
        print(f"Close price data saved to {output_path}")
    elif args.command == "build-minute-cache":
        run_build_minute_cache(args)
    elif args.command == "prepare-data":
        run_data_preparation_flow(args.jobs)
    elif args.command == "train-models":
        run_model_training_flow()
    elif args.command == "performance-test":
        run_performance_test_flow()
    elif args.command == "live-demo":
        run_live_demo_flow()
    else:  # pragma: no cover - defensive
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
