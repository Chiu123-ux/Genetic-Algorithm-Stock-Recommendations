"""GA + GRU 因子策略的回测主流程。

流程概览：按滚动窗口训练预言机 -> 依据预测挑选股票 -> 计算组合收益及风险指标。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.pipelines.gru_forecaster import GRUConfig, GRUFactorForecaster
from src.pipelines.ga_factor_miner import GAFactorMiner, load_ga_config
from src.utils.config_loader import get_config_value
from src.utils.vwap_target_loader import build_target_matrix, load_vwap_target_long


@dataclass
class GRUBacktestConfig:
    """配置 GRU 回测的交易规模、成本以及训练窗口要求。"""
    top_n: int
    initial_capital: float
    transaction_cost: float  # 单边交易成本率 (例如: 0.0015 = 0.15%)
    min_train_samples: int
    train_window_months: int = 18
    retrain_every_n_rebalances: int = 1
    # 支持按月滚动的设置：初始训练窗口（单位月），以及每隔多少月调仓一次
    initial_train_months: int = 12
    rebalance_months: int = 1
    # 向后兼容：如果 rebalance_months 未被使用，仍然可以按交易日频率调仓
    rebalance_frequency: int = 10  # 调仓频率：每N个交易日调仓一次
    # 每隔多少月重新运行 GA（0 表示不在回测中运行 GA）
    run_ga_every_n_months: int = 8
    # 回测中周期性重跑 GA 时使用的轻量参数
    backtest_ga_pop_num: int = 800
    backtest_ga_gen_num: int = 6
    candidate_pool_size: int = 20
    volatility_lookback_days: int = 20
    inverse_vol_weight_power: float = 0.0
    max_single_position_weight: float = 0.15
    enable_market_trend_filter: bool = True
    trend_filter_window: int = 60
    trend_exposure_floor: float = 0.5
    enable_drawdown_scaling: bool = True
    drawdown_scale_trigger: float = 0.12
    drawdown_exposure_floor: float = 0.35
    # 训练集尾部净空（交易日）。0 表示按配置自动推导。
    purge_gap_days: int = 0
    # 收益累计方式：compound(复利, 默认) 或 simple(单利)
    return_calc_mode: str = "compound"
    # 回测收益口径：
    # - daily_nextday: 使用价格序列计算的下一日收益（旧口径）
    # - rebalance_5d: 仅调仓日使用外部目标收益记账（5D事件口径）
    # - daily_external_target: 每日使用外部目标收益记账（适合1D raw return）
    backtest_return_mode: str = "daily_nextday"


def load_gru_backtest_config() -> GRUBacktestConfig:
    """从配置文件读取 GRU 回测参数，并提供合理的默认值。"""
    cfg = get_config_value("gru_pipeline") or {}
    return GRUBacktestConfig(
        top_n=cfg.get("top_n", 10),
        initial_capital=cfg.get("initial_capital", 1_000_000),
        transaction_cost=cfg.get("transaction_cost", 0.0015),  # 提高默认交易成本到0.15%
        min_train_samples=cfg.get("min_train_samples", 60),
        train_window_months=cfg.get("train_window_months", 18),
        retrain_every_n_rebalances=cfg.get("retrain_every_n_rebalances", 1),
        initial_train_months=cfg.get("initial_train_months", 4),
        rebalance_months=cfg.get("rebalance_months", 1),
        rebalance_frequency=cfg.get("rebalance_frequency", 10),  # 默认每10天调仓一次
        run_ga_every_n_months=cfg.get("run_ga_every_n_months", 0),
        backtest_ga_pop_num=cfg.get("backtest_ga_pop_num", 500),
        backtest_ga_gen_num=cfg.get("backtest_ga_gen_num", 5),
        candidate_pool_size=cfg.get("candidate_pool_size", max(cfg.get("top_n", 10) * 2, 20)),
        volatility_lookback_days=cfg.get("volatility_lookback_days", 20),
        inverse_vol_weight_power=cfg.get("inverse_vol_weight_power", 0.0),
        max_single_position_weight=cfg.get("max_single_position_weight", 0.15),
        enable_market_trend_filter=cfg.get("enable_market_trend_filter", True),
        trend_filter_window=cfg.get("trend_filter_window", 60),
        trend_exposure_floor=cfg.get("trend_exposure_floor", 0.5),
        enable_drawdown_scaling=cfg.get("enable_drawdown_scaling", True),
        drawdown_scale_trigger=cfg.get("drawdown_scale_trigger", 0.12),
        drawdown_exposure_floor=cfg.get("drawdown_exposure_floor", 0.35),
        purge_gap_days=cfg.get("purge_gap_days", 0),
        return_calc_mode=str(cfg.get("return_calc_mode", "compound")).lower(),
        backtest_return_mode=str(cfg.get("backtest_return_mode", "daily_nextday")).lower(),
    )


class GRUBacktester:
    """结合 GA 因子与 GRU 预测结果的逐日走滑回测器。"""

    def __init__(
        self,
        base_config: GRUConfig,
        backtest_config: GRUBacktestConfig,
        stock_names: List[str],
        benchmark_series: Dict[str, pd.Series] | None = None,
        benchmark_name: str | None = None,
    ) -> None:
        """保存基础配置并执行关键参数合法性校验。"""
        self.base_config = base_config
        self.config = backtest_config
        self.stock_names = stock_names
        self.benchmark_series = benchmark_series or {}
        self.benchmark_name = benchmark_name or self._infer_primary_benchmark_name()
        if self.config.top_n <= 0:
            raise ValueError("GRU backtest requires top_n > 0")
        if self.config.min_train_samples <= 0:
            raise ValueError("GRU backtest requires min_train_samples > 0")
        if self.config.train_window_months <= 0:
            raise ValueError("GRU backtest requires train_window_months > 0")
        if self.config.retrain_every_n_rebalances <= 0:
            raise ValueError("GRU backtest requires retrain_every_n_rebalances > 0")
        if self.config.return_calc_mode not in {"compound", "simple"}:
            raise ValueError("GRU backtest requires return_calc_mode in {'compound', 'simple'}")
        if self.config.backtest_return_mode not in {"daily_nextday", "rebalance_5d", "daily_external_target"}:
            raise ValueError(
                "GRU backtest requires backtest_return_mode in {'daily_nextday', 'rebalance_5d', 'daily_external_target'}"
            )
        self._effective_purge_gap_days = self._resolve_effective_purge_gap_days()

    def _resolve_effective_purge_gap_days(self) -> int:
        configured_gap = int(getattr(self.config, "purge_gap_days", 0) or 0)
        if configured_gap > 0:
            return configured_gap

        target_mode = str(getattr(self.base_config, "target_mode", "legacy_price_return")).lower()
        if target_mode != "legacy_price_return":
            signal_days = max(0, int(getattr(self.base_config, "signal_to_entry_days", 1) or 0))
            horizon_days = max(0, int(getattr(self.base_config, "target_horizon_days", 0) or 0))
            return signal_days + horizon_days

        return max(0, int(getattr(self.base_config, "predict_days", 0) or 0))

    def _minimum_rebalance_index(self) -> int:
        return int(self.config.min_train_samples) + int(self._effective_purge_gap_days)

    def _infer_primary_benchmark_name(self) -> str | None:
        universe = str(getattr(self.base_config, "backtest_factor_file", "") or getattr(self.base_config, "factor_file", "")).lower()
        available = set(self.benchmark_series)
        if "hs300" in universe and "hs300" in available:
            return "hs300"
        if "zz500" in universe and "zz500" in available:
            return "zz500"
        if "zz500" in available:
            return "zz500"
        if "hs300" in available:
            return "hs300"
        return next(iter(self.benchmark_series), None)

    def _format_rebalance_label(self) -> str:
        if getattr(self.config, "rebalance_months", None):
            return f"每{self.config.rebalance_months}月"
        return f"每{self.config.rebalance_frequency}天"

    def _resolve_training_slice(self, sample_dates: pd.Index, idx: int) -> tuple[int, int]:
        train_end_idx = idx - self._effective_purge_gap_days
        if train_end_idx <= 0:
            return 0, 0

        current_date = pd.Timestamp(sample_dates[train_end_idx - 1])
        window_start_date = current_date - pd.DateOffset(months=int(self.config.train_window_months))
        train_start_idx = int(pd.Index(sample_dates).searchsorted(window_start_date, side="left"))

        if train_end_idx - train_start_idx < self.config.min_train_samples:
            train_start_idx = max(0, train_end_idx - self.config.min_train_samples)

        return train_start_idx, train_end_idx

    def _should_run_periodic_ga(
        self,
        current_date: pd.Timestamp,
        last_ga_refresh_date: pd.Timestamp | None,
    ) -> bool:
        """按真实日期间隔决定是否重跑 GA，而不是按调仓次数计数。"""
        interval_months = int(self.config.run_ga_every_n_months or 0)
        if interval_months <= 0:
            return False
        if last_ga_refresh_date is None:
            return True
        next_refresh_date = pd.Timestamp(last_ga_refresh_date) + pd.DateOffset(months=interval_months)
        return pd.Timestamp(current_date) >= next_refresh_date

    def _build_volatility_frame(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        daily_return_history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if "volatility" in factor_dict:
            # 使用上一交易日已知的波动率，避免将当日信息带入调仓决策
            volatility_df = factor_dict["volatility"].reindex(daily_return_history_df.index).shift(1)
        else:
            lookback = max(5, int(self.config.volatility_lookback_days or 20))
            # 仅使用历史日收益构造滚动波动率，并整体滞后一日，确保 date=t 的权重只依赖 t-1 及更早信息
            volatility_df = daily_return_history_df.rolling(lookback).std().shift(1)
        volatility_df = volatility_df.replace([np.inf, -np.inf], np.nan)
        # 0/负波动率会在逆波动率加权中制造接近无限大的权重，按缺失值处理更稳妥。
        volatility_df = volatility_df.where(volatility_df > 0)

        # 缺失值优先使用同日横截面中位数填充，不跨时间回填，避免未来信息泄露
        row_medians = volatility_df.median(axis=1, skipna=True)
        volatility_df = volatility_df.T.fillna(row_medians).T
        volatility_df = volatility_df.fillna(1.0).clip(lower=1e-6)
        return volatility_df

    def _cap_position_weights(self, weights: pd.Series, total_exposure: float) -> pd.Series:
        if weights.empty or total_exposure <= 0:
            return pd.Series(dtype=float)

        feasible_cap = max(total_exposure / len(weights), 0.0)
        max_weight = max(min(float(self.config.max_single_position_weight), total_exposure), feasible_cap)
        capped = weights.astype(float).copy()
        capped = capped / capped.sum() * total_exposure

        for _ in range(len(capped) + 1):
            over_mask = capped > max_weight + 1e-12
            if not over_mask.any():
                break
            excess = float((capped[over_mask] - max_weight).sum())
            capped.loc[over_mask] = max_weight
            under_mask = ~over_mask
            if not under_mask.any() or excess <= 1e-12:
                break
            under_sum = float(capped[under_mask].sum())
            if under_sum <= 1e-12:
                capped.loc[under_mask] = excess / int(under_mask.sum())
            else:
                capped.loc[under_mask] += excess * capped[under_mask] / under_sum

        total = float(capped.sum())
        if total <= 1e-12:
            return pd.Series(dtype=float)
        return capped / total * total_exposure

    def _compute_target_exposure(self, date_key: pd.Timestamp, capital: float, capital_peak: float) -> Tuple[float, Dict[str, bool]]:
        exposure = 1.0
        flags = {
            "trend_filter_active": False,
            "drawdown_scaling_active": False,
        }

        if self.config.enable_market_trend_filter and self.benchmark_name in self.benchmark_series:
            benchmark_series = self.benchmark_series[self.benchmark_name]
            history = benchmark_series.loc[benchmark_series.index < date_key].dropna().tail(max(5, int(self.config.trend_filter_window or 60)))
            if len(history) >= max(5, int(self.config.trend_filter_window or 60) // 2):
                trend_ma = float(history.mean())
                trend_value = float(history.iloc[-1])
                if trend_value < trend_ma:
                    exposure = min(exposure, float(self.config.trend_exposure_floor))
                    flags["trend_filter_active"] = True

        if self.config.enable_drawdown_scaling and capital_peak > 0:
            drawdown = capital / capital_peak - 1.0
            if drawdown <= -abs(float(self.config.drawdown_scale_trigger)):
                exposure = min(exposure, float(self.config.drawdown_exposure_floor))
                flags["drawdown_scaling_active"] = True

        return max(0.0, min(1.0, exposure)), flags

    def _build_target_weights(
        self,
        pred_series: pd.Series,
        volatility_df: pd.DataFrame | None,
        date_key: pd.Timestamp,
        capital: float,
        capital_peak: float,
    ) -> Tuple[pd.Series, pd.Series, float, Dict[str, bool]]:
        gross_exposure, exposure_flags = self._compute_target_exposure(date_key, capital, capital_peak)
        if pred_series.empty or gross_exposure <= 0:
            return pd.Series(dtype=float), pd.Series(dtype=float), gross_exposure, exposure_flags

        ranked = pred_series.sort_values(ascending=False)
        pool_size = min(len(ranked), max(self.config.top_n, int(self.config.candidate_pool_size or self.config.top_n)))
        candidate_scores = ranked.head(pool_size)

        if volatility_df is not None and date_key in volatility_df.index:
            volatility_slice = volatility_df.loc[date_key].reindex(candidate_scores.index)
        else:
            volatility_slice = pd.Series(1.0, index=candidate_scores.index)

        volatility_slice = volatility_slice.replace([np.inf, -np.inf], np.nan)
        volatility_slice = volatility_slice.fillna(volatility_slice.median())
        volatility_slice = volatility_slice.fillna(1.0).abs().clip(lower=1e-6)

        weight_power = max(0.0, float(self.config.inverse_vol_weight_power or 0.0))
        if weight_power > 0.0:
            risk_adjusted_scores = candidate_scores / np.power(volatility_slice, weight_power)
        else:
            risk_adjusted_scores = candidate_scores.copy()

        selection_names = risk_adjusted_scores.sort_values(ascending=False).head(self.config.top_n).index
        selection_scores = pred_series.reindex(selection_names).sort_values(ascending=False)
        selection_vol = volatility_slice.reindex(selection_scores.index).fillna(volatility_slice.median()).fillna(1.0).clip(lower=1e-6)

        if weight_power > 0.0:
            raw_weights = 1.0 / np.power(selection_vol, weight_power)
        else:
            raw_weights = pd.Series(1.0, index=selection_scores.index)
        raw_weights = raw_weights / raw_weights.sum()
        target_weights = self._cap_position_weights(raw_weights, gross_exposure)
        return target_weights, selection_scores, gross_exposure, exposure_flags

    @staticmethod
    def _compute_performance_metrics(
        curve: pd.Series,
        initial_value: float,
        period_returns: pd.Series | None = None,
        return_calc_mode: str = "compound",
    ) -> Dict[str, float]:
        curve = curve.astype(float)
        if period_returns is None:
            returns = curve.pct_change().fillna(0.0)
        else:
            returns = period_returns.astype(float).fillna(0.0)
        total_return = float(curve.iloc[-1] / initial_value - 1)
        period_days = float((pd.to_datetime(curve.index[-1]) - pd.to_datetime(curve.index[0])).days)
        years = period_days / 365.0 if period_days > 0 else 0.0
        if years > 0:
            if str(return_calc_mode).lower() == "simple":
                annualized_return = float(total_return / years)
            else:
                annualized_return = float((1 + total_return) ** (1 / years) - 1)
        else:
            annualized_return = total_return
        volatility = float(returns.std() * np.sqrt(252))
        cum_max = curve.cummax()
        max_drawdown = float(((curve / cum_max) - 1).min())
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
        }

    def run(self, X: np.ndarray, y: np.ndarray, dates: pd.Index) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """执行滚动回测，返回逐期记录与整体表现摘要。"""
        n_samples = X.shape[0]
        panel_mode = X.ndim == 4 and y.ndim == 2 and X.shape[1] == y.shape[1]
        # 准备日期索引（对应 X/y 的样本序列起点日期）
        dates_dt = pd.to_datetime(dates)

        # 如果配置了按月滚动（推荐），先计算所有调仓的样本索引
        rebalance_indices: List[int] = []
        minimum_rebalance_index = self._minimum_rebalance_index()
        if getattr(self.config, 'rebalance_months', None) and getattr(self.config, 'initial_train_months', None):
            first_date = dates_dt[0]
            last_date = dates_dt[-1]
            # 首次可调仓日期 = 起始日期 + initial_train_months
            next_reb_date = first_date + pd.DateOffset(months=self.config.initial_train_months)
            while next_reb_date <= last_date:
                # 找到第一个大于等于 next_reb_date 的样本索引
                idx = int(dates_dt.searchsorted(next_reb_date))
                if idx >= n_samples:
                    break
                # 确保至少有 min_train_samples 用于训练
                if idx >= minimum_rebalance_index:
                    rebalance_indices.append(idx)
                # 下一个调仓日期按 rebalance_months 增加
                next_reb_date = next_reb_date + pd.DateOffset(months=self.config.rebalance_months)

        # 备用：如果没有计算出按月的调仓点，则沿用按交易日的频率
        if not rebalance_indices:
            start_idx = minimum_rebalance_index
            if start_idx >= n_samples:
                raise ValueError("Dataset is too small for the requested GRU backtest window")
            # 生成按日计数的调仓索引（首次在 start_idx，之后每 rebalance_frequency 天）
            rebalance_indices = list(range(start_idx, n_samples, self.config.rebalance_frequency))
        else:
            start_idx = rebalance_indices[0]

        target_mode = str(getattr(self.base_config, "target_mode", "legacy_price_return")).lower()
        external_target_enabled = target_mode != "legacy_price_return"
        use_external_rebalance_return = external_target_enabled and self.config.backtest_return_mode == "rebalance_5d"
        use_external_daily_return = external_target_enabled and self.config.backtest_return_mode == "daily_external_target"

        # 载入价格数据以便计算真实的逐日收益（用于回测逐日更新）
        data_forecaster = GRUFactorForecaster(self.base_config)
        try:
            price_df_for_backtest, factor_dict_for_backtest = data_forecaster.load_data()
            daily_return_history_df = price_df_for_backtest.pct_change()
            daily_return_history_df = daily_return_history_df.replace([np.inf, -np.inf], np.nan)
            # 下一日收益：pct_change 然后向上移动一行，使得 index=t 对应 t->t+1 的收益
            one_day_returns_df = price_df_for_backtest.pct_change().shift(-1)
            one_day_returns_df = one_day_returns_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            volatility_df = self._build_volatility_frame(factor_dict_for_backtest, daily_return_history_df)
        except Exception:
            # 如果无法加载价格，退回到只使用 y 中提供的目标（旧行为）
            one_day_returns_df = None
            volatility_df = None

        external_target_returns_df: pd.DataFrame | None = None
        if use_external_rebalance_return or use_external_daily_return:
            try:
                field_name = str(getattr(self.base_config, "target_backtest_field", "raw_vwap_5d"))
                backtest_parquet_root = (
                    getattr(self.base_config, "target_backtest_parquet_root", None)
                    or getattr(self.base_config, "target_parquet_root", None)
                )
                long_df = load_vwap_target_long(
                    backtest_parquet_root,
                    required_columns=("date", "stock", field_name),
                )
                external_target_returns_df = build_target_matrix(long_df, field_name)
                external_target_returns_df = external_target_returns_df.reindex(columns=self.stock_names)
            except Exception as exc:
                print(f"⚠️ 外部目标回测标签加载失败，将退回逐日收益模式: {exc}")
                use_external_rebalance_return = False
                use_external_daily_return = False

        capital = self.config.initial_capital
        capital_peak = capital
        is_compound_mode = self.config.return_calc_mode == "compound"
        benchmark_values = {
            name: float(self.config.initial_capital)
            for name in self.benchmark_series
        }
        records: List[Dict[str, object]] = []
        
        # 持仓跟踪
        current_positions: List[str] = []  # 当前持仓股票
        current_weights: Dict[str, float] = {}
        last_rebalance_idx = start_idx  # 上次调仓时点
        current_exposure = 0.0
        current_exposure_flags = {
            "trend_filter_active": False,
            "drawdown_scaling_active": False,
        }
        current_turnover = 0.0
        
        # 计算预期的调仓次数用于进度显示
        expected_rebalances = len(rebalance_indices)
        rebalance_count = 0
        freq_note = f"{self._format_rebalance_label()}调仓"
        print(
            f"开始回测: 共{n_samples}个样本, 首次训练窗结束样本索引 {start_idx}, "
            f"预计调仓{expected_rebalances}次, {freq_note}, purge_gap={self._effective_purge_gap_days}天"
        )

        # 使用 while 循环以便在回测过程中重新构建数据集（例如每隔 N 月运行 GA）
        idx = start_idx
        pbar = tqdm(total=n_samples - start_idx, desc="回测进度", unit="天")
        
        daily_alphas = {}  # 记录每日预测 alpha
        forecaster = None  # 跨周期保持对上一个预言机的引用
        model_label = str(getattr(self.base_config, "model_type", "gru") or "gru").strip().upper()
        initial_ga_refresh_date = getattr(self.config, "initial_ga_refresh_date", None)
        last_ga_refresh_date: pd.Timestamp | None = pd.to_datetime(initial_ga_refresh_date) if initial_ga_refresh_date is not None else None
        last_train_rebalance_count: int | None = None

        while idx < n_samples:
            should_rebalance = (idx in rebalance_indices) or (not current_positions)
            date_key = pd.to_datetime(dates[idx])

            if should_rebalance:
                rebalance_count += 1
                tqdm.write(f"  调仓 {rebalance_count}/{expected_rebalances}: 样本索引 {idx}")
                force_retrain = forecaster is None

                # 如果配置中需要在回测中定期运行 GA，则按真实月份间隔触发
                if self._should_run_periodic_ga(date_key, last_ga_refresh_date):
                    if last_ga_refresh_date is None:
                        tqdm.write("    开始运行 GA 挖因子（首次调仓初始化）...")
                    else:
                        tqdm.write(
                            f"    开始运行 GA 挖因子（距离上次 GA 已满{self.config.run_ga_every_n_months}个月）..."
                        )
                    # 记录当前的基准日期，便于在重建数据集后对齐索引
                    old_date = pd.to_datetime(dates[idx])
                    ga_cfg = load_ga_config()
                    # 将 GA 的预测天数与回测使用的目标对齐
                    ga_cfg.predict_days = self.base_config.predict_days
                    miner = GAFactorMiner(ga_cfg)
                    try:
                        # 在回测中使用轻量级 GA 参数以加速（smoke-run 模式）
                        miner.mine(
                            pop_num=self.config.backtest_ga_pop_num,
                            gen_num=self.config.backtest_ga_gen_num,
                            cutoff_date=old_date,
                        )
                        last_ga_refresh_date = old_date
                        force_retrain = True
                        tqdm.write("    GA 执行完成，已更新 generated_factors_latest.txt")
                    except Exception as exc:
                        tqdm.write(f"    警告：GA 在回测中执行失败：{exc}")

                    # 重新构建训练集与目标（因为因子表达式已更新）
                    # 预测目标天数已经在 base_config 中配置
                    forecaster_reload = GRUFactorForecaster(self.base_config)
                    price_df, factor_dict = forecaster_reload.load_data()
                    expressions = forecaster_reload.load_factor_expressions()
                    X_new, y_new, dates_new = forecaster_reload.create_dataset(price_df, factor_dict, expressions)
                    X, y, dates = X_new, y_new, dates_new
                    n_samples = X.shape[0]
                    dates_dt = pd.to_datetime(dates)
                    # 重新计算调仓索引（与最初计算逻辑一致）
                    rebalance_indices = []
                    minimum_rebalance_index = self._minimum_rebalance_index()
                    if getattr(self.config, 'rebalance_months', None) and getattr(self.config, 'initial_train_months', None):
                        first_date = dates_dt[0]
                        last_date = dates_dt[-1]
                        next_reb_date = first_date + pd.DateOffset(months=self.config.initial_train_months)
                        while next_reb_date <= last_date:
                            ridx = int(dates_dt.searchsorted(next_reb_date))
                            if ridx >= n_samples:
                                break
                            if ridx >= minimum_rebalance_index:
                                rebalance_indices.append(ridx)
                            next_reb_date = next_reb_date + pd.DateOffset(months=self.config.rebalance_months)
                    if not rebalance_indices:
                        start_idx = minimum_rebalance_index
                        rebalance_indices = list(range(start_idx, n_samples, self.config.rebalance_frequency))
                    expected_rebalances = len(rebalance_indices)
                    # 将 idx 对齐到重建后的日期位置（以之前记录的日期为基准）
                    idx = int(dates_dt.searchsorted(old_date))
                    if idx >= n_samples:
                        # 如果重建后索引超出范围，结束回测循环
                        tqdm.write("    重建数据后索引超出范围，终止回测")
                        break

                should_retrain_model = force_retrain
                if not should_retrain_model and last_train_rebalance_count is not None:
                    should_retrain_model = (rebalance_count - last_train_rebalance_count) >= self.config.retrain_every_n_rebalances

                if should_retrain_model:
                    tqdm.write(f"    执行 {model_label} 重训")
                    if forecaster is not None:
                        del forecaster
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    train_start_idx, train_end_idx = self._resolve_training_slice(dates, idx)
                    if train_end_idx - train_start_idx <= 0:
                        raise ValueError(
                            "No trainable samples available after applying purge gap; "
                            "increase min_train_samples or reduce purge_gap_days."
                        )
                    tqdm.write(
                        f"    训练窗口: {pd.Timestamp(dates[train_start_idx]).date()} -> {pd.Timestamp(dates[train_end_idx - 1]).date()} "
                        f"({train_end_idx - train_start_idx}个样本, 最近{self.config.train_window_months}个月)"
                    )
                    forecaster = GRUFactorForecaster(self.base_config)
                    forecaster.train(
                        X[train_start_idx:train_end_idx],
                        y[train_start_idx:train_end_idx],
                        sample_dates=dates[train_start_idx:train_end_idx],
                        save_artifacts=False,
                    )
                    last_train_rebalance_count = rebalance_count
                else:
                    assert last_train_rebalance_count is not None
                    tqdm.write(
                        f"    复用上次 {model_label} 模型: 距上次重训 {rebalance_count - last_train_rebalance_count} 次调仓, "
                        f"配置为每 {self.config.retrain_every_n_rebalances} 次调仓重训一次"
                    )

                if forecaster is None:
                    raise RuntimeError(f"{model_label} forecaster is unavailable at rebalance time")

                predictions = forecaster.predict_sequences(X[idx : idx + 1])[0]

                pred_series = pd.Series(predictions, index=self.stock_names)
                if panel_mode:
                    pred_series = pred_series.where(pd.Series(np.isfinite(y[idx]), index=self.stock_names))
                pred_series = pred_series.replace([np.inf, -np.inf], np.nan).dropna()
                
                # 记录当天的预测 alpha 评分
                daily_alphas[dates[idx]] = pred_series.copy()

                target_weights, selection, gross_exposure, exposure_flags = self._build_target_weights(
                    pred_series,
                    volatility_df,
                    date_key,
                    capital,
                    capital_peak,
                )
                new_positions = selection.index.tolist()
                prev_weights = pd.Series(current_weights, dtype=float)
                turnover_ratio = 0.5 * float(target_weights.sub(prev_weights, fill_value=0.0).abs().sum())
                transaction_cost_applied = self.config.transaction_cost * turnover_ratio
                
                # 计算持仓变化和交易成本
                if current_positions:
                    position_changes = len(set(new_positions) - set(current_positions))
                    tqdm.write(f"    持仓调整: {position_changes}/{self.config.top_n}只股票变化, 换手率{turnover_ratio:.1%}")
                else:
                    tqdm.write(f"    初次建仓: 选中{len(new_positions)}只股票")
                    
                tqdm.write(f"    选中股票: {', '.join(new_positions[:3])}{'...' if len(new_positions) > 3 else ''}")
                if exposure_flags["trend_filter_active"]:
                    tqdm.write(f"    趋势过滤生效: 仓位降至{gross_exposure:.0%}")
                if exposure_flags["drawdown_scaling_active"]:
                    tqdm.write(f"    回撤降仓生效: 仓位降至{gross_exposure:.0%}")
                tqdm.write(f"    预期收益: {selection.mean():.4f}, 交易成本: {transaction_cost_applied:.4f}, 目标仓位: {gross_exposure:.0%}")
                tqdm.write("")  # 空行分隔
                    
                current_positions = new_positions
                current_weights = target_weights.to_dict()
                last_rebalance_idx = idx
                current_exposure = float(target_weights.sum()) if not target_weights.empty else 0.0
                current_exposure_flags = exposure_flags
                current_turnover = turnover_ratio
                mean_prediction = float((selection * target_weights.reindex(selection.index).fillna(0.0)).sum()) if not selection.empty else 0.0
            else:
                # 不调仓，使用上期持仓
                transaction_cost_applied = 0.0  # 不调仓则无交易成本
                current_turnover = 0.0
                mean_prediction = 0.0  # 无新预测
                
                # 为了满足导师需要，每天输出股票的 alpha 预测值
                # 直接调用上次训练好的预言机对当天数据做推理（不重训、不调仓）
                if forecaster is not None:
                    try:
                        with torch.no_grad():
                            predictions = forecaster.predict_sequences(X[idx : idx + 1])[0]
                        pred_series = pd.Series(predictions, index=self.stock_names)
                        if panel_mode:
                            pred_series = pred_series.where(pd.Series(np.isfinite(y[idx]), index=self.stock_names))
                        pred_series = pred_series.replace([np.inf, -np.inf], np.nan).dropna()
                        daily_alphas[dates[idx]] = pred_series.copy()
                    except Exception:
                        pass

            # 计算当期收益（基于当前持仓）
            if use_external_rebalance_return:
                # 仅在调仓日记账：使用 entry_date 的 raw_vwap_5d 加权收益
                if should_rebalance and current_weights and external_target_returns_df is not None:
                    try:
                        actual_series = external_target_returns_df.loc[date_key]
                        if isinstance(actual_series, pd.Series):
                            actual_series = actual_series.reindex(self.stock_names).fillna(0.0)
                        else:
                            actual_series = pd.Series(actual_series, index=self.stock_names).fillna(0.0)
                    except Exception as exc:
                        print(f"⚠️ 无法提取 {date_key} 的 raw_vwap_5d 收益: {exc}，本次收益记为0。")
                        actual_series = pd.Series(0.0, index=self.stock_names)

                    weight_series = pd.Series(current_weights, dtype=float)
                    realised = actual_series.reindex(weight_series.index).fillna(0.0)
                    mean_realised = float((realised * weight_series).sum())
                    traded = True
                else:
                    mean_realised = 0.0
                    traded = should_rebalance and transaction_cost_applied > 0.0
                period_return = mean_realised - transaction_cost_applied if traded else 0.0
            elif use_external_daily_return:
                # 每日记账：使用外部目标收益（例如 raw1d）按当前持仓加权
                if current_weights and external_target_returns_df is not None:
                    try:
                        actual_series = external_target_returns_df.loc[date_key]
                        if isinstance(actual_series, pd.Series):
                            actual_series = actual_series.reindex(self.stock_names).fillna(0.0)
                        else:
                            actual_series = pd.Series(actual_series, index=self.stock_names).fillna(0.0)
                    except Exception as exc:
                        print(f"⚠️ 无法提取 {date_key} 的外部目标收益: {exc}，当日收益记为0。")
                        actual_series = pd.Series(0.0, index=self.stock_names)

                    weight_series = pd.Series(current_weights, dtype=float)
                    realised = actual_series.reindex(weight_series.index).fillna(0.0)
                    mean_realised = float((realised * weight_series).sum())
                    # 每日都记收益；交易成本仍只在调仓日产生
                    period_return = mean_realised - transaction_cost_applied
                else:
                    mean_realised = 0.0
                    period_return = -transaction_cost_applied if should_rebalance and transaction_cost_applied > 0.0 else 0.0
            else:
                # 逐日收益模式（兼容旧行为）
                if one_day_returns_df is not None:
                    # dates 中的元素应该与 price_df 的索引对齐，尝试逐日取值
                    try:
                        actual_series = one_day_returns_df.loc[date_key]
                        # 如果取出来是 DataFrame/Series，保证为 Series 并与 stock_names 对齐
                        if isinstance(actual_series, pd.Series):
                            actual_series = actual_series.reindex(self.stock_names).fillna(0.0)
                        else:
                            actual_series = pd.Series(actual_series, index=self.stock_names).fillna(0.0)
                    except Exception as e:
                        # 如果无法定位当日期真实收益，且此时 y 可能是标准化过的多日收益，绝不能直接用来复利！
                        print(f"⚠️ 无法提取 {date_key} 的真实收益: {e}，当日真实收益记为 0以防计算崩溃。")
                        actual_series = pd.Series(0.0, index=self.stock_names)
                else:
                    print(f"⚠️ 遇到严重错误：当日价格序列表不存在，真实收益记为 0。")
                    actual_series = pd.Series(0.0, index=self.stock_names)

                if current_weights:
                    weight_series = pd.Series(current_weights, dtype=float)
                    realised = actual_series.reindex(weight_series.index).fillna(0.0)
                    mean_realised = float((realised * weight_series).sum())
                    traded = True
                else:
                    mean_realised = 0.0
                    traded = transaction_cost_applied > 0.0

                # 计算净收益
                period_return = mean_realised - transaction_cost_applied if traded else 0.0
            period_return = max(period_return, -0.99)  # guard against numerical blow-up
            if is_compound_mode:
                capital *= 1 + period_return
            else:
                capital += self.config.initial_capital * period_return
            capital_peak = max(capital_peak, capital)

            benchmark_returns: Dict[str, float] = {}
            for benchmark_name, benchmark_series in self.benchmark_series.items():
                bench_return = 0.0
                try:
                    benchmark_value = benchmark_series.loc[date_key]
                    prev_idx = benchmark_series.index.get_loc(date_key)
                    if isinstance(prev_idx, slice):
                        prev_idx = prev_idx.start
                    if isinstance(prev_idx, (int, np.integer)) and prev_idx > 0:
                        prev_value = benchmark_series.iloc[prev_idx - 1]
                        if pd.notna(prev_value) and float(prev_value) != 0.0 and pd.notna(benchmark_value):
                            bench_return = float(benchmark_value / prev_value - 1)
                except Exception:
                    bench_return = 0.0
                benchmark_returns[benchmark_name] = bench_return
                if is_compound_mode:
                    benchmark_values[benchmark_name] *= 1 + bench_return
                else:
                    benchmark_values[benchmark_name] += self.config.initial_capital * bench_return

            primary_benchmark_return = benchmark_returns.get(self.benchmark_name or "", 0.0)
            primary_benchmark_value = benchmark_values.get(self.benchmark_name or "", self.config.initial_capital)
            if is_compound_mode:
                if primary_benchmark_return <= -0.999999:
                    excess_return = 0.0
                else:
                    excess_return = float((1.0 + period_return) / (1.0 + primary_benchmark_return) - 1.0)
            else:
                excess_return = float(period_return - primary_benchmark_return)

            record = {
                "date": dates[idx],
                "predicted_mean": mean_prediction,
                "realised_mean": mean_realised,
                "net_return": period_return,
                "portfolio_value": capital,
                "selected": ",".join(current_positions),
                "rebalanced": should_rebalance or idx == start_idx,
                "transaction_cost": transaction_cost_applied,
                "turnover": current_turnover,
                "gross_exposure": current_exposure,
                "trend_filter_active": current_exposure_flags["trend_filter_active"],
                "drawdown_scaling_active": current_exposure_flags["drawdown_scaling_active"],
                "benchmark_name": self.benchmark_name,
                "benchmark_return": primary_benchmark_return,
                "benchmark_value": primary_benchmark_value,
                "excess_return": excess_return,
            }
            for benchmark_name, bench_value in benchmark_values.items():
                record[f"benchmark_return_{benchmark_name}"] = benchmark_returns.get(benchmark_name, 0.0)
                record[f"benchmark_value_{benchmark_name}"] = bench_value
            records.append(record)

            # 迭代推进与进度条更新
            pbar.update(1)
            idx += 1

        pbar.close()
        
        # 保存日度 alpha 预测信息到 CSV，不影响系统正常返回值
        try:
            if daily_alphas:
                alpha_df = pd.DataFrame(daily_alphas).T  # 行是日期，列是股票代码
                alpha_df.index.name = "date"
                out_dir = getattr(self.base_config, "log_output_dir", "./reports")
                os.makedirs(out_dir, exist_ok=True)
                timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(out_dir, f"daily_alphas_{timestamp_str}.csv")
                alpha_df.to_csv(out_path)
                print(f"✅ [新增功能] 日度股票预测 Alpha 矩阵已导出至: {out_path}")
        except Exception as e:
            print(f"⚠️ 导出日度 Alpha 矩阵时出错: {e}")

        result = pd.DataFrame(records)
        if result.empty:
            raise ValueError("GRU backtest produced no records; check configuration")

        values = result["portfolio_value"].astype(float)
        result["portfolio_curve"] = values / self.config.initial_capital
        performance = self._compute_performance_metrics(
            values.set_axis(pd.to_datetime(result["date"])),
            self.config.initial_capital,
            period_returns=result["net_return"],
            return_calc_mode=self.config.return_calc_mode,
        )
        period_days = float((pd.to_datetime(result["date"]).iloc[-1] - pd.to_datetime(result["date"]).iloc[0]).days)

        primary_benchmark_metrics: Dict[str, float] = {}
        if "benchmark_value" in result:
            benchmark_curve = result["benchmark_value"].astype(float)
            result["benchmark_curve"] = benchmark_curve / self.config.initial_capital
            primary_benchmark_metrics = self._compute_performance_metrics(
                benchmark_curve.set_axis(pd.to_datetime(result["date"])),
                self.config.initial_capital,
                period_returns=result["benchmark_return"],
                return_calc_mode=self.config.return_calc_mode,
            )
            if is_compound_mode:
                safe_benchmark_curve = result["benchmark_curve"].replace(0.0, np.nan)
                result["excess_curve"] = result["portfolio_curve"].astype(float) / safe_benchmark_curve
            else:
                result["excess_curve"] = 1.0 + (
                    (result["portfolio_curve"].astype(float) - 1.0)
                    - (result["benchmark_curve"].astype(float) - 1.0)
                )
        else:
            result["benchmark_curve"] = np.nan
            result["excess_curve"] = np.nan

        benchmark_summaries: Dict[str, Dict[str, float]] = {}
        for benchmark_name in self.benchmark_series:
            value_col = f"benchmark_value_{benchmark_name}"
            if value_col not in result:
                continue
            curve = result[value_col].astype(float)
            result[f"benchmark_curve_{benchmark_name}"] = curve / self.config.initial_capital
            benchmark_summaries[benchmark_name] = self._compute_performance_metrics(
                curve.set_axis(pd.to_datetime(result["date"])),
                self.config.initial_capital,
                period_returns=result[f"benchmark_return_{benchmark_name}"],
                return_calc_mode=self.config.return_calc_mode,
            )

        summary = {
            "start_date": str(result["date"].iloc[0]),
            "end_date": str(result["date"].iloc[-1]),
            "period_days": period_days,
            "initial_capital": self.config.initial_capital,
            "total_return": performance["total_return"],
            "annualized_return": performance["annualized_return"],
            "volatility": performance["volatility"],
            "max_drawdown": performance["max_drawdown"],
            "return_calc_mode": self.config.return_calc_mode,
            "benchmark_name": self.benchmark_name,
            "benchmark_metrics": primary_benchmark_metrics,
            "benchmark_metrics_by_name": benchmark_summaries,
        }

        if primary_benchmark_metrics:
            summary["benchmark_total_return"] = primary_benchmark_metrics["total_return"]
            summary["excess_total_return"] = float(result["excess_curve"].iloc[-1] - 1)
            summary["excess_return_spread"] = float(
                performance["total_return"] - primary_benchmark_metrics["total_return"]
            )

        # 显示回测完成总结
        print("\n" + "="*60)
        print("📊 回测完成总结")
        print("="*60)
        print(f"📅 回测期间: {result['date'].iloc[0]} 至 {result['date'].iloc[-1]} ({int(period_days)}天)")
        print(f"💰 初始资金: {self.config.initial_capital:,.0f}")
        print(f"💰 最终资金: {values.iloc[-1]:,.0f}")
        print(f"🧮 收益累计: {'复利' if is_compound_mode else '单利'} ({self.config.return_calc_mode})")
        print(f"📈 总收益率: {performance['total_return']:+.2%}")
        print(f"📈 年化收益: {performance['annualized_return']:+.2%}")
        print(f"📊 波动率: {performance['volatility']:.2%}")
        print(f"📉 最大回撤: {performance['max_drawdown']:.2%}")
        if primary_benchmark_metrics:
            print(f"📎 基线: {self.benchmark_name}, 收益率 {primary_benchmark_metrics['total_return']:+.2%}")
            print(f"🟡 超额收益: {summary['excess_total_return']:+.2%}")
            print(f"🟠 收益差: {summary['excess_return_spread']:+.2%}")
        print(f"🔄 总调仓次数: {rebalance_count}")
        print(f"⚙️  调仓频率: {self._format_rebalance_label()}")
        print("="*60 + "\n")

        return result, summary
