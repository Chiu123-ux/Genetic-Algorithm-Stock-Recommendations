"""基于遗传程序算法的因子挖掘主流程。

核心步骤：读取配置与训练数据 -> 筛选基础因子 -> 调用自定义 gplearn
进行符号回归搜索 -> 输出带评分的因子表达式文件。
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from numpy.lib.format import open_memmap
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
import time

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover - torch 可选依赖
    torch = None

from gplearn.functions import _function_map
from gplearn.typed_functions import TYPED_FUNCTION_MAP
from gplearn.typed_functions import DAILY_MATRIX, MINUTE_TENSOR  # noqa: E402
from toolkit.setupGPlearn import my_gplearn

from src.utils.config_loader import get_config_value
from src.utils.data_io import (
    load_minute_supported_stock_columns,
    load_price_and_factors,
    restrict_price_factors_to_columns,
)
from src.utils.minute_tensor_cache import get_cache_file_path, load_minute_day_cache, load_minute_day_cache_axes
from src.utils.vwap_target_loader import build_target_matrix, load_vwap_target_long
@dataclass
class GAFactorConfig:
    """GA 因子挖掘流程的核心参数集合。"""
    num_factors: int
    train_months: int
    predict_days: int
    output_directory: str
    price_file: str
    factor_file: str
    max_factors: int = 25  # 添加最大因子数参数，默认为25
    # 基础日频终端选择模式：auto=按类别自动挑选，all=使用全部可用日频字段，configured=仅使用 daily_field_names
    daily_field_selection_mode: str = "auto"
    # 显式指定参与 GA 的日频字段名列表（字段名应与 factors bundle 中的键一致）
    daily_field_names: Tuple[str, ...] | None = None
    # 惩罚因子多样性相关性的强度（0=no penalty, 1=强烈惩罚）
    diversity_penalty_alpha: float = 0.3
    # warm-start 的种子表达式文件（可选）
    seed_expression_file: str | None = None
    # 最多读取的种子表达式数量（避免特征爆炸）
    max_seed_expressions: int = 5
    # 是否将 seed 直接注入 GP 初始种群
    inject_seed_programs: bool = False
    # 每个 seed 额外生成多少个轻微扰动副本加入第 0 代
    seed_program_mutation_copies: int = 2
    # 生成扰动副本时的点变异节点替换概率
    seed_program_mutation_replace_rate: float = 0.15
    # GA 训练并行 worker 数（传递给 gplearn 的 n_jobs）
    ga_workers: int = 50
    # GA 进化超参数（mine 未显式传参时使用）
    ga_pop_num: int = 500
    ga_gen_num: int = 6
    ga_p_crossover: float = 0.8
    ga_p_subtree_mutation: float = 0.05
    ga_p_hoist_mutation: float = 0.05
    ga_p_point_mutation: float = 0.05
    # 是否启用 GA 评分阶段的 GPU 加速（需安装 torch 且 CUDA 可用）
    ga_use_gpu: bool = True
    # GA 评分阶段使用的 CUDA 设备号
    ga_cuda_device: int = 0
    # 适应度函数指标，可选 'mean_abs_ic', 'icir', 'mean_ic', 'signed_mean_ic'
    fitness_metric: str = "mean_abs_ic"
    # 因子硬筛选：最终得分低于该阈值直接剔除
    ga_min_ic_threshold: float = 0.0
    # 至少需要多少个有效 IC 交易日，过少则视为不可靠因子
    ga_min_valid_ic_days: int = 0
    # 因子硬筛选：与已选因子的最大相关性超过该阈值直接剔除
    ga_max_similarity_threshold: float = 1.0
    # 风险调整后的组合评估参数
    ga_top_n: int = 10
    ga_candidate_pool_size: int = 20
    ga_inverse_vol_weight_power: float = 0.0
    ga_downside_penalty: float = 0
    ga_drawdown_penalty: float = 0
    ga_volatility_penalty: float = 0
    ga_random_seed_base: int = 42
    ga_iteration_multiplier: int = 2
    # GA 最大迭代轮数；为 None 时按 num_factors * ga_iteration_multiplier 推导
    ga_max_iterations: int | None = None
    ga_factors_per_iteration: int = 1
    # 单轮候选池放大倍数：候选数 = ga_factors_per_iteration * ga_round_candidate_multiplier
    ga_round_candidate_multiplier: int = 3
    # 频率覆盖配额：纯日频、纯分钟衍生、混合频率的最少入选数量
    ga_min_pure_daily: int = 0
    ga_min_pure_minute_derived: int = 0
    ga_min_mixed_frequency: int = 0
    # 当 mixed 配额未满足时，是否允许自动合成 mixed 候选（默认关闭，仅做实验用途）
    ga_enable_mixed_synthesis_fallback: bool = False
    # 若为 True，最终配额未满足时报错；否则仅告警
    ga_frequency_quota_hard: bool = False
    ga_resume_cache_hours: float = 1.0
    ga_volatility_window_min: int = 5
    ga_volatility_window_max: int = 20
    # 目标标签来源: legacy_price_return 或 external_* (例如 external_vwap_5d/external_raw1d)
    target_mode: str = "legacy_price_return"
    # 外部标签 parquet 路径（支持单文件或目录分片）
    target_parquet_root: str | None = None
    # 外部标签训练字段（neu_all_vwap_5d / vwap_5d）
    target_train_field: str = "neu_all_vwap_5d"
    # 信号到入场交易日偏移
    signal_to_entry_days: int = 1
    # 目标持有期（仅用于元信息，不参与外部标签计算）
    target_horizon_days: int = 5
    # 函数库配置：minimal / balanced / aggressive
    function_library_profile: str = "balanced"
    # 可选：直接指定函数白名单（优先级高于 profile）
    function_set: Tuple[str, ...] | None = None
    # 是否启用时间函数（delayy/delta/decayl/rankk 等）
    enable_time_functions: bool = True
    # 是否启用 WQ 风格时间函数（固定窗口 ts_*_5/10/20）
    enable_wq_time_functions: bool = False
    # 是否启用逻辑门控函数（gtpn/andpn/...）
    enable_logic_functions: bool = False
    # 是否启用三角函数（sin/cos/tan）
    enable_trig_functions: bool = False
    # 是否启用 3D-native 分钟+日频混合挖掘
    enable_3d_mining: bool = False
    # 分钟 cache 根目录（由 build-minute-cache 生成）
    minute_cache_root: str | None = None
    # 参与 3D 挖掘的分钟字段列表
    minute_field_names: Tuple[str, ...] | None = None
    # 分钟字段名前缀，避免与日频字段重名
    minute_field_prefix: str = "min_"
    # 3D 特征张量是否使用 memmap
    ga_3d_use_memmap: bool = True
    # 3D memmap 输出目录
    ga_3d_memmap_dir: str | None = None
    # 3D 张量 dtype
    ga_3d_dtype: str = "float32"
    # 3D 张量构建：按交易日并行读取 minute cache 的 worker 数（1=串行）
    ga_3d_loader_workers: int = 1
    # 3D 张量构建：读取任务预取队列上限（避免一次性积压过多内存）
    ga_3d_loader_prefetch: int = 2
    # 3D 张量构建：归一化阶段并行 worker 数（1=串行）
    ga_3d_normalize_workers: int = 1
    # 3D 分钟模式下，日频特征是否滞后 signal_to_entry_days 天。
    # 默认 False：当前 3D smoke 按 T 日收盘后出信号来解释，因此允许使用 T 日日频信息。
    # 若要改成严格的盘中信号语义（例如 10:00 仅用当时可见信息），可显式设为 True。
    ga_3d_shift_daily_features: bool = False


@dataclass
class FactorExpression:
    """承载单个符号表达式及其评估得分。"""
    expression: str
    score: float


@dataclass
class _RoundCandidate:
    """单轮候选的评分结果与预测缓存。"""
    expression: str
    score: float
    prediction: np.ndarray
    frequency_type: str


class GAFactorMiner:
    """利用自定义 gplearn 工具寻找表现优秀的组合因子表达式。"""

    _FUNCTION_GROUPS: Dict[str, Tuple[str, ...]] = {
        "basic": (
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "abs",
            "neg",
            "inv",
            "max",
            "min",
        ),
        "time": (
            "delayy",
            "delta",
            "signedpower",
            "decayl",
            "stdd",
            "rankk",
        ),
        "wq_time": (
            "ts_min_5",
            "ts_min_10",
            "ts_min_20",
            "ts_max_5",
            "ts_max_10",
            "ts_max_20",
            "ts_argmax_5",
            "ts_argmax_10",
            "ts_argmax_20",
            "ts_argmin_5",
            "ts_argmin_10",
            "ts_argmin_20",
            "ts_rank_5",
            "ts_rank_10",
            "ts_rank_20",
            "ts_sum_5",
            "ts_sum_10",
            "ts_sum_20",
            "ts_product_5",
            "ts_product_10",
            "ts_product_20",
            "ts_stddev_5",
            "ts_stddev_10",
            "ts_stddev_20",
        ),
        "logic": (
            "gtpn",
            "andpn",
            "orpn",
            "ltpn",
            "gtp",
            "andp",
            "orp",
            "ltp",
            "gtn",
            "andn",
            "orn",
            "ltn",
        ),
        "trig": (
            "sin",
            "cos",
            "tan",
        ),
        "typed_basic": (
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "abssqrt",
            "log",
            "abs",
            "neg",
            "inv",
            "sigmoid",
            "tanh",
            "max",
            "min",
            "sign",
            "diff",
            "zscore",
            "signedpower",
            "ts_delay_5",
            "ts_delay_10",
            "ts_delay_20",
            "ts_delta_5",
            "ts_delta_10",
            "ts_delta_20",
            "ts_mean_5",
            "ts_mean_10",
            "ts_mean_20",
            "ts_std_5",
            "ts_std_10",
            "ts_std_20",
        ),
        "typed_daily": (
            "day_delay_5",
            "day_delay_10",
            "day_delay_20",
            "day_delta_5",
            "day_delta_10",
            "day_delta_20",
            "day_mean_5",
            "day_mean_10",
            "day_mean_20",
            "day_std_5",
            "day_std_10",
            "day_std_20",
            "d_abs",
            "d_log",
            "d_sqrt",
            "d_sign",
            "d_add",
            "d_sub",
            "d_mul",
            "d_div",
            "d_max",
            "d_min",
            "d_absdiff",
            "d_ratio",
            "d_norm_spread",
        ),
        "typed_reducer": (
            "nanmean",
            "nanstd",
            "nanmax",
            "nanmin",
            "sum",
            "last",
            "range",
            "argmax",
            "argmin",
            "q10",
            "q50",
            "q90",
            "am_mean",
            "am_std",
            "am_range",
            "pm_mean",
            "pm_std",
            "pm_range",
            "mid_mean",
            "mid_std",
            "mid_range",
            "open15_mean",
            "open15_std",
            "open15_range",
            "open30_mean",
            "open30_std",
            "open30_sum",
            "open30_range",
            "close15_mean",
            "close15_std",
            "close15_range",
            "close30_mean",
            "close30_std",
            "close30_sum",
            "close30_range",
            "am_pm_spread",
            "midday_reversal",
            "cov",
            "corr",
            "beta",
            "am_beta",
            "pm_beta",
            "weighted_mean",
            "mean_spread",
        ),
    }

    def __init__(self, config: GAFactorConfig) -> None:
        self.config = config
        self._training_feature_names: List[str] = []
        self._training_daily_feature_count: int = 0
        self._training_minute_feature_count: int = 0
        self._external_target_cache: dict[str, pd.DataFrame] = {}
        self.function_set = self._resolve_function_set()

    def _resolve_function_set(self) -> Tuple[str, ...]:
        """根据配置构建函数白名单，并过滤掉未注册函数。"""
        logger = logging.getLogger(__name__)
        is_3d_mode = bool(getattr(self.config, "enable_3d_mining", False))

        configured_set = getattr(self.config, "function_set", None)
        if configured_set:
            chosen = [str(name).strip() for name in configured_set if str(name).strip()]
        elif is_3d_mode:
            chosen = list(self._FUNCTION_GROUPS["typed_basic"])
            chosen.extend(self._FUNCTION_GROUPS["typed_daily"])
            chosen.extend(self._FUNCTION_GROUPS["typed_reducer"])
        else:
            profile = str(getattr(self.config, "function_library_profile", "balanced")).strip().lower()
            chosen = list(self._FUNCTION_GROUPS["basic"])

            if profile in {"balanced", "aggressive"} and bool(getattr(self.config, "enable_time_functions", True)):
                chosen.extend(self._FUNCTION_GROUPS["time"])
            if profile in {"balanced", "aggressive"} and bool(getattr(self.config, "enable_wq_time_functions", False)):
                chosen.extend(self._FUNCTION_GROUPS["wq_time"])
            if profile == "aggressive" and bool(getattr(self.config, "enable_logic_functions", False)):
                chosen.extend(self._FUNCTION_GROUPS["logic"])
            if profile == "aggressive" and bool(getattr(self.config, "enable_trig_functions", False)):
                chosen.extend(self._FUNCTION_GROUPS["trig"])

        deduped: List[str] = []
        for name in chosen:
            if name not in deduped:
                deduped.append(name)

        function_map = TYPED_FUNCTION_MAP if is_3d_mode else _function_map
        valid = tuple(name for name in deduped if name in function_map)
        dropped = [name for name in deduped if name not in function_map]
        if dropped:
            logger.warning(f"忽略未注册函数: {dropped}")
        if not valid:
            raise ValueError("GA function_set resolved to empty list")
        return valid

    def _minute_terminal_name(self, field_name: str) -> str:
        prefix = str(getattr(self.config, "minute_field_prefix", "min_") or "min_")
        return f"{prefix}{field_name}"

    def _minute_field_names(self) -> List[str]:
        raw_fields = getattr(self.config, "minute_field_names", None)
        if not raw_fields:
            return []
        return [str(name).strip() for name in raw_fields if str(name).strip()]

    def _select_3d_terminal_names(self, factors: Dict[str, pd.DataFrame]) -> tuple[List[str], List[str], List[str]]:
        daily_feature_names = self._select_base_factors(factors)
        minute_feature_names = self._minute_field_names()
        if not minute_feature_names:
            raise ValueError("enable_3d_mining requires minute_field_names to be configured")
        terminal_names = list(daily_feature_names) + [self._minute_terminal_name(name) for name in minute_feature_names]
        return terminal_names, daily_feature_names, minute_feature_names

    def _build_3d_training_arrays(
        self,
        price_df: pd.DataFrame,
        factors: Dict[str, pd.DataFrame],
        daily_feature_names: List[str],
        minute_feature_names: List[str],
        cutoff_date: pd.Timestamp | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray, np.ndarray, List[str]]:
        logger = logging.getLogger(__name__)
        total_start_ts = time.perf_counter()
        if not self.config.minute_cache_root:
            raise ValueError("enable_3d_mining requires minute_cache_root to be configured")

        end_date = price_df.index[-1]
        if cutoff_date is not None:
            cutoff_ts = pd.to_datetime(cutoff_date)
            end_date = min(end_date, cutoff_ts)
        start_date = end_date - pd.DateOffset(months=self.config.train_months)
        price_train = price_df.loc[start_date:end_date]
        if price_train.empty:
            raise ValueError("训练窗口为空：请检查 cutoff_date 与 train_months 配置")

        target_mode = str(getattr(self.config, "target_mode", "legacy_price_return")).lower()
        signal_to_entry_days = max(0, int(getattr(self.config, "signal_to_entry_days", 1) or 0))
        external_target_enabled = target_mode != "legacy_price_return"
        if external_target_enabled:
            if signal_to_entry_days >= len(price_train.index):
                raise ValueError("Not enough rows in training window for signal_to_entry_days")
            feature_index = price_train.index[:-signal_to_entry_days] if signal_to_entry_days > 0 else price_train.index
            entry_index = price_train.index[signal_to_entry_days:] if signal_to_entry_days > 0 else price_train.index
            target_matrix = self._load_external_target_matrix(self.config.target_train_field)
            target_matrix = target_matrix.reindex(columns=price_train.columns)
            returns = target_matrix.reindex(entry_index)
            returns.index = feature_index
        else:
            forward = price_train.shift(-self.config.predict_days)
            returns = (forward - price_train) / price_train
            returns = returns.iloc[:-self.config.predict_days]

        cache_root = Path(self.config.minute_cache_root)
        candidate_dates = list(pd.to_datetime(returns.index))
        available_dates: List[pd.Timestamp] = []
        reference_minutes: List[str] | None = None
        axis_scan_start_ts = time.perf_counter()
        for trade_date in candidate_dates:
            cache_file = get_cache_file_path(cache_root, trade_date)
            if not cache_file.exists():
                continue
            try:
                cache_axes = load_minute_day_cache_axes(cache_root, trade_date)
            except Exception as exc:
                logger.warning(f"读取分钟 cache 失败，跳过 {trade_date.date()}: {exc}")
                continue

            minute_labels = [str(item) for item in cache_axes["minutes"].tolist()]
            cache_fields = {str(item) for item in cache_axes["fields"].tolist()}
            if any(field not in cache_fields for field in minute_feature_names):
                logger.warning(f"分钟 cache 缺少字段，跳过 {trade_date.date()}: required={minute_feature_names}, available={sorted(cache_fields)}")
                continue
            if reference_minutes is None:
                reference_minutes = minute_labels
            elif minute_labels != reference_minutes:
                logger.warning(f"分钟轴不一致，跳过 {trade_date.date()}")
                continue
            available_dates.append(trade_date)
        axis_scan_cost = time.perf_counter() - axis_scan_start_ts

        if not available_dates or reference_minutes is None:
            raise ValueError("训练窗口内没有可用的分钟 cache 日期")

        returns = returns.reindex(available_dates)
        returns = returns.reindex(columns=price_train.columns)
        n_dates = len(available_dates)
        n_stocks = returns.shape[1]
        n_minutes = len(reference_minutes)
        terminal_names = list(daily_feature_names) + [self._minute_terminal_name(name) for name in minute_feature_names]
        n_features = len(terminal_names)
        dtype = np.dtype(str(getattr(self.config, "ga_3d_dtype", "float32") or "float32"))
        shift_daily_features = bool(getattr(self.config, "ga_3d_shift_daily_features", True))

        daily_frames: List[pd.DataFrame] = []
        for name in daily_feature_names:
            frame = factors[name]
            if shift_daily_features and signal_to_entry_days > 0:
                frame = frame.shift(signal_to_entry_days)
            frame = frame.reindex(index=returns.index, columns=returns.columns)
            frame = frame.replace([np.inf, -np.inf], np.nan)
            med = frame.median(axis=0, skipna=True)
            frame = frame.fillna(med.to_dict()).fillna(0.0)
            daily_frames.append(frame.astype(dtype, copy=False))

        if "volatility" in factors:
            volatility_frame = factors["volatility"]
            if shift_daily_features and signal_to_entry_days > 0:
                volatility_frame = volatility_frame.shift(signal_to_entry_days)
            volatility_frame = volatility_frame.reindex(index=returns.index, columns=returns.columns)
        else:
            rolling_window_min = max(1, int(getattr(self.config, "ga_volatility_window_min", 5) or 1))
            rolling_window_max = max(rolling_window_min, int(getattr(self.config, "ga_volatility_window_max", 20) or rolling_window_min))
            rolling_window = max(rolling_window_min, min(rolling_window_max, int(self.config.predict_days or rolling_window_max)))
            volatility_frame = price_train.pct_change().rolling(rolling_window).std()
            if shift_daily_features and signal_to_entry_days > 0:
                volatility_frame = volatility_frame.shift(signal_to_entry_days)
            volatility_frame = volatility_frame.reindex(index=returns.index, columns=returns.columns)

        volatility_frame = volatility_frame.replace([np.inf, -np.inf], np.nan)
        vol_col_median = volatility_frame.median(axis=0, skipna=True)
        volatility_frame = volatility_frame.fillna(vol_col_median.to_dict())
        volatility_frame = volatility_frame.fillna(float(np.nanmedian(volatility_frame.values)) if np.isfinite(np.nanmedian(volatility_frame.values)) else 1.0)
        volatility_frame = volatility_frame.clip(lower=1e-6)

        use_memmap = bool(getattr(self.config, "ga_3d_use_memmap", True))
        cache_mmap_mode = "r" if use_memmap else None
        if use_memmap:
            memmap_root = Path(getattr(self.config, "ga_3d_memmap_dir", None) or (Path(self.config.output_directory) / "_ga_3d_memmap"))
            memmap_root.mkdir(parents=True, exist_ok=True)
            memmap_path = memmap_root / f"training_tensor_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
            factor_array = open_memmap(memmap_path, mode="w+", dtype=dtype, shape=(n_dates, n_stocks, n_minutes, n_features))
            factor_array[:] = np.nan
        else:
            factor_array = np.full((n_dates, n_stocks, n_minutes, n_features), np.nan, dtype=dtype)

        stock_codes = returns.columns.astype(str)
        stock_to_target_pos = {code: index for index, code in enumerate(stock_codes)}
        availability_mask = np.zeros((n_dates, n_stocks), dtype=bool)

        for feature_index, frame in enumerate(daily_frames):
            factor_array[:, :, :, feature_index] = frame.to_numpy(dtype=dtype, copy=False)[:, :, None]

        def _populate_one_date(date_index: int, trade_date: pd.Timestamp, cache_payload: dict[str, np.ndarray]) -> None:
            cache_stocks = [str(item) for item in cache_payload["stocks"].tolist()]
            cache_fields = [str(item) for item in cache_payload["fields"].tolist()]
            cache_data = np.asarray(cache_payload["data"], dtype=dtype)
            cache_data = np.where(np.isfinite(cache_data), cache_data, np.nan)
            cache_field_pos = {name: idx for idx, name in enumerate(cache_fields)}
            source_pos_map = {code: idx for idx, code in enumerate(cache_stocks)}
            intersect_codes = [code for code in stock_codes if code in source_pos_map]
            if not intersect_codes:
                return
            source_rows = np.array([source_pos_map[code] for code in intersect_codes], dtype=np.int64)
            target_rows = np.array([stock_to_target_pos[code] for code in intersect_codes], dtype=np.int64)
            availability_mask[date_index, target_rows] = True
            for offset, minute_field_name in enumerate(minute_feature_names):
                feature_pos = len(daily_feature_names) + offset
                factor_array[date_index, target_rows, :, feature_pos] = cache_data[source_rows, :, cache_field_pos[minute_field_name]]

        loader_workers = max(1, int(getattr(self.config, "ga_3d_loader_workers", 1) or 1))
        loader_prefetch = max(
            loader_workers,
            int(getattr(self.config, "ga_3d_loader_prefetch", loader_workers * 2) or (loader_workers * 2)),
        )

        fill_start_ts = time.perf_counter()
        if loader_workers <= 1 or n_dates <= 1:
            for date_index, trade_date in enumerate(available_dates):
                cache_payload = load_minute_day_cache(cache_root, trade_date, mmap_mode=cache_mmap_mode)
                _populate_one_date(date_index, trade_date, cache_payload)
        else:
            print(
                "[3D-BUILD] parallel minute cache loading: "
                f"workers={loader_workers}, prefetch={loader_prefetch}, dates={n_dates}"
            )

            with ThreadPoolExecutor(max_workers=loader_workers) as executor:
                pending: dict = {}
                next_index = 0

                def _submit_more() -> None:
                    nonlocal next_index
                    while next_index < n_dates and len(pending) < loader_prefetch:
                        idx = next_index
                        trade_date = available_dates[idx]
                        future = executor.submit(load_minute_day_cache, cache_root, trade_date, cache_mmap_mode)
                        pending[future] = (idx, trade_date)
                        next_index += 1

                _submit_more()
                while pending:
                    done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        date_index, trade_date = pending.pop(future)
                        try:
                            cache_payload = future.result()
                        except Exception as exc:
                            raise RuntimeError(f"加载分钟 cache 失败: {trade_date.date()} -> {exc}") from exc
                        _populate_one_date(date_index, trade_date, cache_payload)
                    _submit_more()
        fill_cost = time.perf_counter() - fill_start_ts

        normalize_start_ts = time.perf_counter()
        self._normalize_3d_feature_tensor_in_place(factor_array)
        normalize_cost = time.perf_counter() - normalize_start_ts

        returns_matrix = returns.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        close_valid_frame = factors.get("close", price_df)
        if shift_daily_features and signal_to_entry_days > 0:
            close_valid_frame = close_valid_frame.shift(signal_to_entry_days)
        close_valid_frame = close_valid_frame.reindex(index=returns.index, columns=returns.columns)
        close_valid_matrix = close_valid_frame.replace([np.inf, -np.inf], np.nan).notna().to_numpy(dtype=bool)
        close_nonzero_matrix = close_valid_frame.fillna(0.0).ne(0.0).to_numpy(dtype=bool)
        valid_mask = np.isfinite(returns_matrix) & availability_mask & close_valid_matrix & close_nonzero_matrix
        volatility_matrix = volatility_frame.to_numpy(dtype=float)
        volatility_matrix = np.nan_to_num(volatility_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        volatility_matrix = np.clip(volatility_matrix, 1e-6, None)
        self._training_feature_names = terminal_names
        total_cost = time.perf_counter() - total_start_ts
        print(
            "[3D-BUILD] done: "
            f"dates={n_dates} stocks={n_stocks} minutes={n_minutes} features={n_features} "
            f"| axis_scan={axis_scan_cost:.2f}s fill={fill_cost:.2f}s normalize={normalize_cost:.2f}s total={total_cost:.2f}s"
        )
        return factor_array, returns_matrix, (n_dates, n_stocks), valid_mask, volatility_matrix, terminal_names

    def _normalize_3d_feature_tensor_in_place(self, factor_array: np.ndarray) -> None:
        n_features = factor_array.shape[-1]

        def _normalize_one_feature(feature_index: int) -> None:
            feature_slice = factor_array[..., feature_index]
            finite_mask = np.isfinite(feature_slice)
            if not finite_mask.any():
                feature_slice[...] = 0.0
                return

            finite_values = feature_slice[finite_mask].astype(np.float64, copy=False)
            mean_value = float(finite_values.mean())
            std_value = float(finite_values.std())
            if not np.isfinite(mean_value):
                mean_value = 0.0
            if not np.isfinite(std_value) or std_value <= 1e-12:
                std_value = 1.0

            feature_slice[finite_mask] = (feature_slice[finite_mask] - mean_value) / std_value
            feature_slice[~finite_mask] = 0.0

        normalize_workers = max(1, int(getattr(self.config, "ga_3d_normalize_workers", 1) or 1))
        if normalize_workers <= 1 or n_features <= 1:
            for feature_index in range(n_features):
                _normalize_one_feature(feature_index)
            return

        worker_count = min(normalize_workers, n_features)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(_normalize_one_feature, range(n_features)))

    def _resolve_min_valid_ic_days(self, n_rows: int) -> int:
        configured = int(getattr(self.config, "ga_min_valid_ic_days", 0) or 0)
        if configured > 0:
            return configured
        return max(5, int(np.ceil(max(1, n_rows) * 0.02)))

    @staticmethod
    def _reduce_ic_values(ic_array: np.ndarray, metric_type: str) -> float:
        ic_array = np.asarray(ic_array, dtype=float)
        if ic_array.size == 0:
            return 0.0

        metric = str(metric_type or "mean_abs_ic").strip().lower()
        if metric == "icir":
            ic_mean = np.mean(ic_array)
            ic_std = np.std(ic_array, ddof=1)
            return float(abs(ic_mean) / (ic_std + 1e-6))
        if metric == "mean_ic":
            return float(abs(np.mean(ic_array)))
        if metric == "signed_mean_ic":
            return float(np.mean(ic_array))
        return float(np.mean(np.abs(ic_array)))

    def _load_external_target_matrix(self, field_name: str) -> pd.DataFrame:
        """Load and cache one external target field matrix (date x stock)."""
        if field_name in self._external_target_cache:
            return self._external_target_cache[field_name]
        if not self.config.target_parquet_root:
            raise ValueError("target_parquet_root is required when target_mode is external target mode")

        long_df = load_vwap_target_long(
            self.config.target_parquet_root,
            required_columns=("date", "stock", field_name),
        )
        matrix = build_target_matrix(long_df, field_name)
        self._external_target_cache[field_name] = matrix
        return matrix

    @staticmethod
    def _split_arguments(argument_block: str) -> List[str]:
        """解析函数参数列表，处理括号嵌套。"""
        args: List[str] = []
        depth = 0
        start = 0
        for index, char in enumerate(argument_block):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(argument_block[start:index].strip())
                start = index + 1
        final = argument_block[start:].strip()
        if final:
            args.append(final)
        return args

    def _parse_expression(self, expression: str):
        """递归解析符号表达式。"""
        expr = expression.strip()
        if not expr:
            raise ValueError("Empty expression")

        try:
            float(expr)
            return ("const", expr)
        except ValueError:
            pass

        if "(" not in expr:
            return ("var", expr)
        if not expr.endswith(")"):
            raise ValueError(f"Invalid expression format: {expression}")

        func_name_end = expr.index("(")
        func_name = expr[:func_name_end].strip()
        inner = expr[func_name_end + 1 : -1]
        arguments = [self._parse_expression(item) for item in self._split_arguments(inner)]
        return ("func", func_name, arguments)

    def _evaluate_node(
        self,
        node,
        data_sources: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, pd.Index, pd.Index]:
        """计算解析树节点。"""
        kind = node[0]
        if kind == "var":
            name = node[1]
            if name not in data_sources:
                raise KeyError(name)
            frame = data_sources[name]
            return frame.values, frame.index, frame.columns

        if kind == "const":
            value = float(node[1])
            sample_frame = next(iter(data_sources.values()))
            shape = sample_frame.values.shape
            constant_array = np.full(shape, value)
            return constant_array, sample_frame.index, sample_frame.columns

        func_name = node[1]
        children = node[2]
        if func_name not in _function_map:
            raise KeyError(func_name)
        evaluated = [self._evaluate_node(child, data_sources) for child in children]
        arrays = [item[0] for item in evaluated]
        result = _function_map[func_name](*arrays)
        index = evaluated[0][1]
        columns = evaluated[0][2]
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(result), index, columns

    def _evaluate_expression(
        self,
        expression: str,
        data_sources: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """求值表达式，返回 DataFrame。"""
        node = self._parse_expression(expression)
        values, index, columns = self._evaluate_node(node, data_sources)
        return pd.DataFrame(values, index=index, columns=columns)

    def _load_seed_expressions(self) -> List[str]:
        """从配置的种子表达式文件中读取 warm-start 初始表达式。"""
        seed_file = self.config.seed_expression_file
        if not seed_file:
            return []

        path = Path(seed_file)
        if not path.exists():
            logging.getLogger(__name__).warning(f"Seed expression file not found: {path}")
            return []

        expressions: List[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            expr = line
            if ":" in expr:
                expr = expr.split(":", 1)[1].strip()
            if "#" in expr:
                expr = expr.split("#", 1)[0].strip()
            if expr:
                expressions.append(expr)

        max_count = max(int(getattr(self.config, "max_seed_expressions", 5) or 0), 0)
        if max_count > 0:
            expressions = expressions[:max_count]
        return expressions

    def _collect_expression_variables_from_node(self, node) -> Set[str]:
        kind = node[0]
        if kind == "var":
            return {node[1]}
        if kind == "const":
            return set()

        variables: Set[str] = set()
        for child in node[2]:
            variables.update(self._collect_expression_variables_from_node(child))
        return variables

    def _collect_expression_variables(self, expression: str) -> Set[str]:
        node = self._parse_expression(expression)
        return self._collect_expression_variables_from_node(node)

    @staticmethod
    def _empty_frequency_counter() -> Dict[str, int]:
        return {
            "pure_daily": 0,
            "pure_minute_derived": 0,
            "mixed_frequency": 0,
            "unknown": 0,
        }

    def _classify_expression_frequency(
        self,
        expression: str,
        daily_feature_names: Set[str],
    ) -> str:
        """按表达式中的终端变量来源分类频率形态。"""
        minute_prefix = str(getattr(self.config, "minute_field_prefix", "min_") or "min_")
        try:
            variables = self._collect_expression_variables(expression)
        except Exception:
            return "unknown"

        if not variables:
            # 常数表达式默认归为日频
            return "pure_daily"

        has_minute = False
        has_daily = False
        for name in variables:
            if str(name).startswith(minute_prefix):
                has_minute = True
            elif name in daily_feature_names:
                has_daily = True
            else:
                # 未知变量按日频终端处理，避免把合法日频误判成 unknown
                has_daily = True

        if has_minute and has_daily:
            return "mixed_frequency"
        if has_minute:
            return "pure_minute_derived"
        if has_daily:
            return "pure_daily"
        return "unknown"

    def _extend_feature_names_for_seed_expressions(
        self,
        feature_names: List[str],
        factors: Dict[str, pd.DataFrame],
        seed_expressions: List[str],
    ) -> List[str]:
        if not seed_expressions:
            return feature_names

        extended = list(feature_names)
        for expression in seed_expressions:
            try:
                for variable_name in sorted(self._collect_expression_variables(expression)):
                    if variable_name in factors and variable_name not in extended:
                        extended.append(variable_name)
            except Exception:
                continue
        return extended

    def _expression_node_to_program(self, node, feature_index_map: Dict[str, int]):
        kind = node[0]
        if kind == "var":
            name = node[1]
            if name not in feature_index_map:
                raise KeyError(name)
            return [feature_index_map[name]]

        if kind == "const":
            return [float(node[1])]

        func_name = node[1]
        if func_name not in _function_map:
            raise KeyError(func_name)
        if func_name not in self.function_set:
            raise ValueError(f"Seed function '{func_name}' is not enabled in GP function_set")

        program = [_function_map[func_name]]
        for child in node[2]:
            program.extend(self._expression_node_to_program(child, feature_index_map))
        return program

    def _build_seed_programs(
        self,
        seed_expressions: List[str],
        feature_names: List[str],
    ) -> List[list]:
        if not seed_expressions:
            return []

        logger = logging.getLogger(__name__)
        feature_index_map = {name: index for index, name in enumerate(feature_names)}
        programs: List[list] = []
        for index, expression in enumerate(seed_expressions, start=1):
            try:
                node = self._parse_expression(expression)
                missing = sorted(
                    name for name in self._collect_expression_variables_from_node(node)
                    if name not in feature_index_map
                )
                if missing:
                    raise KeyError(", ".join(missing))
                programs.append(self._expression_node_to_program(node, feature_index_map))
            except Exception as exc:
                logger.warning(f"Skip GP seed program #{index} due to compile error: {exc}")
        return programs

    def _iter_round_candidates(self, model) -> List[object]:
        """返回单轮 GA 可供筛选的候选程序，优先使用最终代种群。"""
        candidates_per_iteration = max(1, int(getattr(self.config, "ga_factors_per_iteration", 1) or 1))
        candidate_multiplier = max(1, int(getattr(self.config, "ga_round_candidate_multiplier", 3) or 3))
        population_history = getattr(model, "_programs", None)
        if isinstance(population_history, list) and population_history:
            final_generation = population_history[-1] or []
            metric = getattr(model, "_metric", None)
            greater_is_better = bool(getattr(metric, "greater_is_better", True))

            def _fitness_key(program) -> float:
                raw_fitness = getattr(program, "raw_fitness_", None)
                if raw_fitness is None or not np.isfinite(raw_fitness):
                    return float("-inf") if greater_is_better else float("inf")
                return float(raw_fitness)

            ranked_programs = [program for program in final_generation if program is not None]
            ranked_programs.sort(key=_fitness_key, reverse=greater_is_better)
            if ranked_programs:
                return ranked_programs[: max(candidates_per_iteration * candidate_multiplier, candidates_per_iteration)]

        best_program = getattr(model, "_program", None)
        if best_program is not None:
            return [best_program]
        return [model]

    @staticmethod
    def _candidate_expression(candidate) -> str:
        return str(candidate)

    @staticmethod
    def _candidate_predict(model, candidate, x_array: np.ndarray) -> np.ndarray:
        if hasattr(candidate, "execute"):
            return np.asarray(candidate.execute(x_array))
        return np.asarray(model.predict(x_array))

    def load_training_data(self) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """加载价格与原始因子数据，为后续训练做准备。"""
        price_df, factor_dict = load_price_and_factors(self.config.price_file, self.config.factor_file)
        if bool(getattr(self.config, "enable_3d_mining", False)) and self.config.minute_cache_root:
            minute_supported = load_minute_supported_stock_columns(self.config.minute_cache_root)
            original_count = price_df.shape[1]
            price_df, factor_dict = restrict_price_factors_to_columns(price_df, factor_dict, minute_supported)
            reduced_count = price_df.shape[1]
            if reduced_count < original_count:
                print(
                    f"[3D-UNIVERSE] restricted stock columns to minute-cache intersection: "
                    f"{original_count} -> {reduced_count}"
                )
        return price_df, factor_dict

    def _select_base_factors(self, factors: Dict[str, pd.DataFrame]) -> List[str]:
        """从不同类型的基础因子中挑选出核心特征。"""
        logger = logging.getLogger(__name__)
        minute_prefix = str(getattr(self.config, "minute_field_prefix", "min_") or "min_")
        factor_keys = [name for name in factors.keys() if not str(name).startswith(minute_prefix)]
        if not factor_keys:
            raise ValueError("No daily factor keys available for GA feature selection")

        selection_mode = str(getattr(self.config, "daily_field_selection_mode", "auto") or "auto").strip().lower()
        if selection_mode not in {"auto", "all", "configured"}:
            logger.warning(f"未知 daily_field_selection_mode={selection_mode}，回退为 auto")
            selection_mode = "auto"

        configured_names = [
            str(name).strip()
            for name in (getattr(self.config, "daily_field_names", None) or ())
            if str(name).strip()
        ]
        resolved_configured = [name for name in configured_names if name in factors and name in factor_keys]
        missing_configured = [name for name in configured_names if name not in factors]
        if missing_configured:
            logger.warning(f"忽略缺失的日频字段: {missing_configured}")

        if selection_mode == "configured":
            if not resolved_configured:
                raise ValueError("daily_field_selection_mode=configured but no configured daily fields are available")
            return resolved_configured

        if selection_mode == "all":
            return resolved_configured + [name for name in factor_keys if name not in resolved_configured]

        upper_name_map = {name: name.upper() for name in factor_keys}

        def _by_keywords(keywords: Tuple[str, ...]) -> Tuple[str, ...]:
            return tuple(
                name
                for name in factor_keys
                if any(keyword in upper_name_map[name] for keyword in keywords)
            )

        categories: Dict[str, Iterable[str]] = {
            # 优先使用当前回测数据中的字段命名，同时兼容历史字段
            "price": ("close_adj", "open_adj", "high_adj", "low_adj", "close", "open", "high", "low"),
            "volume": _by_keywords(("VOLUME", "AMT", "DEALNUM", "VOL")),
            "momentum": ("pct_chg", "turn", "free_turn", "turnover_rate"),
            "valuation": ("pe_ttm", "ps_ttm", "s_val_pb_new", "pcf_ocf_ttm", "free_mkt_cap", "mkt_cap_ard", "total_mv", "circ_mv"),
            "risk_quality": ("volatility", "roe_ttm", "bps"),
            "money_flow": _by_keywords(
                (
                    "MFD",
                    "MONEYFLOW",
                    "INFLOW",
                    "BUY_VALUE",
                    "SELL_VALUE",
                    "BUY_VOLUME",
                    "SELL_VOLUME",
                    "VALUE_DIFF",
                    "VOLUME_DIFF",
                    "TRADES_",
                )
            ),
        }

        selected: List[str] = []
        for names in categories.values():
            for name in names:
                # 逐类尝试加入因子，确保覆盖多维信息
                if name in factors and name not in selected:
                    selected.append(name)
                # 使用配置中的最大因子数限制
                if len(selected) >= self.config.max_factors:
                    return selected

        # 若分类优先后仍未达到上限，则按剩余字段补齐，尽量让 GA 用到更广的基础因子
        for name in factor_keys:
            if name not in selected:
                selected.append(name)
            if len(selected) >= self.config.max_factors:
                break
        return selected

    def _build_training_arrays(
        self,
        price_df: pd.DataFrame,
        factors: Dict[str, pd.DataFrame],
        feature_names: List[str],
        seed_expressions: List[str] | None = None,
        cutoff_date: pd.Timestamp | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
        """构建训练矩阵并处理缺失值，返回扁平化后的特征与收益。"""
        if bool(getattr(self.config, "enable_3d_mining", False)):
            terminal_names, daily_feature_names, minute_feature_names = self._select_3d_terminal_names(factors)
            x_array, y_matrix, shape, valid_mask, volatility_matrix, resolved_names = self._build_3d_training_arrays(
                price_df=price_df,
                factors=factors,
                daily_feature_names=daily_feature_names,
                minute_feature_names=minute_feature_names,
                cutoff_date=cutoff_date,
            )
            self._training_feature_names = resolved_names or terminal_names
            self._training_daily_feature_count = len(daily_feature_names)
            self._training_minute_feature_count = len(minute_feature_names)
            return x_array, y_matrix, shape, valid_mask, volatility_matrix

        seed_expressions = seed_expressions or []
        if len(feature_names) == 0 and len(seed_expressions) == 0:
            raise ValueError("No base features and no seed expressions were provided.")

        end_date = price_df.index[-1]
        if cutoff_date is not None:
            cutoff_ts = pd.to_datetime(cutoff_date)
            end_date = min(end_date, cutoff_ts)
        start_date = end_date - pd.DateOffset(months=self.config.train_months)
        price_train = price_df.loc[start_date:end_date]
        if price_train.empty:
            raise ValueError("训练窗口为空：请检查 cutoff_date 与 train_months 配置")

        target_mode = str(getattr(self.config, "target_mode", "legacy_price_return")).lower()
        signal_to_entry_days = max(0, int(getattr(self.config, "signal_to_entry_days", 1) or 0))
        external_target_enabled = target_mode != "legacy_price_return"
        if external_target_enabled:
            if signal_to_entry_days >= len(price_train.index):
                raise ValueError("Not enough rows in training window for signal_to_entry_days")
            feature_index = price_train.index[:-signal_to_entry_days] if signal_to_entry_days > 0 else price_train.index
            entry_index = price_train.index[signal_to_entry_days:] if signal_to_entry_days > 0 else price_train.index
            target_matrix = self._load_external_target_matrix(self.config.target_train_field)
            target_matrix = target_matrix.reindex(columns=price_train.columns)
            returns = target_matrix.reindex(entry_index)
            returns.index = feature_index
        else:
            # 利用前移价格计算未来收益，作为回归目标
            forward = price_train.shift(-self.config.predict_days)
            returns = (forward - price_train) / price_train
            returns = returns.iloc[:-self.config.predict_days]

        if "volatility" in factors:
            volatility_frame = factors["volatility"].reindex(index=returns.index, columns=returns.columns)
        else:
            rolling_window_min = max(1, int(getattr(self.config, "ga_volatility_window_min", 5) or 1))
            rolling_window_max = max(rolling_window_min, int(getattr(self.config, "ga_volatility_window_max", 20) or rolling_window_min))
            rolling_window = max(rolling_window_min, min(rolling_window_max, int(self.config.predict_days or rolling_window_max)))
            volatility_frame = price_train.pct_change().rolling(rolling_window).std()
            volatility_frame = volatility_frame.reindex(index=returns.index, columns=returns.columns)

        volatility_frame = volatility_frame.replace([np.inf, -np.inf], np.nan)
        vol_col_median = volatility_frame.median(axis=0, skipna=True)
        volatility_frame = volatility_frame.fillna(vol_col_median.to_dict())
        volatility_frame = volatility_frame.fillna(float(np.nanmedian(volatility_frame.values)) if np.isfinite(np.nanmedian(volatility_frame.values)) else 1.0)
        volatility_frame = volatility_frame.clip(lower=1e-6)

        feature_frames: List[pd.DataFrame] = []
        resolved_feature_names: List[str] = []
        for name in feature_names:
            # 使用 reindex 保证对齐且不会抛 KeyError；保留 NaN 以便后续填充策略
            frame = factors[name].reindex(index=returns.index, columns=returns.columns)
            frame = frame.replace([np.inf, -np.inf], np.nan)

            # 按列（每只股票）用中位数填充缺失值，若列全为 NaN 再填 0
            med = frame.median(axis=0, skipna=True)
            # med -> dict 保证按列填充
            frame = frame.fillna(med.to_dict()).fillna(0.0)

            # 按列标准化，避免不同量纲导致的极端值（std 为 0 时替换为 1）
            col_mean = frame.mean(axis=0)
            col_std = frame.std(axis=0).replace(0, 1.0)
            frame = (frame - col_mean) / col_std

            feature_frames.append(frame)
            resolved_feature_names.append(name)

        # warm-start：将种子表达式先计算为特征，拼接到训练输入
        if seed_expressions:
            logger = logging.getLogger(__name__)
            data_sources = dict(factors)
            successful_count = 0
            for index, expression in enumerate(seed_expressions, start=1):
                try:
                    frame = self._evaluate_expression(expression, data_sources)
                    frame = frame.reindex(index=returns.index, columns=returns.columns)
                    frame = frame.replace([np.inf, -np.inf], np.nan)

                    med = frame.median(axis=0, skipna=True)
                    frame = frame.fillna(med.to_dict()).fillna(0.0)

                    col_mean = frame.mean(axis=0)
                    col_std = frame.std(axis=0).replace(0, 1.0)
                    frame = (frame - col_mean) / col_std

                    feature_frames.append(frame)
                    resolved_feature_names.append(f"seed_expr_{index:02d}")
                    successful_count += 1
                except Exception as exc:
                    logger.warning(f"Skip seed expression #{index} due to evaluation error: {exc}")
            logger.info(
                f"Warm-start seed expressions loaded: {successful_count}/{len(seed_expressions)}"
            )

        if len(feature_frames) == 0:
            raise ValueError("No valid feature frames were built from base factors and seed expressions.")

        self._training_feature_names = resolved_feature_names

        # 将所有特征合并为三维数组 (n_dates, n_stocks, n_features)
        factor_array = np.stack([f.values for f in feature_frames], axis=-1)
        n_dates, n_stocks, n_features = factor_array.shape
        # 安全兜底：确保数值有限
        factor_array = np.nan_to_num(factor_array, nan=0.0, posinf=0.0, neginf=0.0)

        returns_matrix = returns.replace([np.inf, -np.inf], np.nan).values
        valid_mask = np.isfinite(returns_matrix)
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        volatility_matrix = volatility_frame.values.astype(float)
        volatility_matrix = np.nan_to_num(volatility_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        volatility_matrix = np.clip(volatility_matrix, 1e-6, None)

        X = factor_array.reshape(n_dates * n_stocks, n_features)
        y = returns_matrix.reshape(n_dates * n_stocks)
        valid_flat = valid_mask.reshape(n_dates * n_stocks)

        return X, y, (n_dates, n_stocks), valid_flat, volatility_matrix

    def _calculate_signed_ic(
        self, 
        x_array: np.ndarray, 
        y_array: np.ndarray, 
        valid_flat: np.ndarray, 
        n_stocks: int
    ) -> float:
        """计算带符号的IC值用于最终排序"""
        # 目前未使用：保留占位或未来实现
        return 0.0

    def mine(
        self,
        pop_num: int | None = None,
        gen_num: int | None = None,
        cutoff_date: pd.Timestamp | str | None = None,
    ) -> List[FactorExpression]:
        pop_num = int(pop_num if pop_num is not None else getattr(self.config, "ga_pop_num", 50))
        gen_num = int(gen_num if gen_num is not None else getattr(self.config, "ga_gen_num", 1))
        stage_start_ts = time.perf_counter()
        price_df, factor_dict = self.load_training_data()
        print(f"[GA-STAGE] load_training_data done in {time.perf_counter() - stage_start_ts:.2f}s")
        is_3d_mode = bool(getattr(self.config, "enable_3d_mining", False))
        seed_expressions = [] if is_3d_mode else self._load_seed_expressions()
        feature_names = self._select_base_factors(factor_dict)
        if not is_3d_mode:
            feature_names = self._extend_feature_names_for_seed_expressions(
                feature_names,
                factor_dict,
                seed_expressions,
            )
        stage_start_ts = time.perf_counter()
        x_array, y_array, shape, valid_flat, volatility_matrix = self._build_training_arrays(
            price_df,
            factor_dict,
            feature_names,
            seed_expressions=seed_expressions,
            cutoff_date=cutoff_date,
        )
        print(f"[GA-STAGE] build_training_arrays done in {time.perf_counter() - stage_start_ts:.2f}s")
        model_feature_names = self._training_feature_names if self._training_feature_names else feature_names
        seed_programs = [] if is_3d_mode else self._build_seed_programs(seed_expressions, model_feature_names)
        n_dates, n_stocks = shape

        expressions: List[FactorExpression] = []
        # 存储已选择因子在训练集上的预测向量，用于计算与新候选因子的相关性惩罚
        selected_preds_list: List[np.ndarray] = []

        logger = logging.getLogger(__name__)

        use_gpu_requested = bool(getattr(self.config, "ga_use_gpu", True))
        cuda_index = int(getattr(self.config, "ga_cuda_device", 0) or 0)
        ga_workers = max(1, int(getattr(self.config, "ga_workers", 1)))
        inject_seed_programs = bool(getattr(self.config, "inject_seed_programs", True))
        seed_program_mutation_copies = max(0, int(getattr(self.config, "seed_program_mutation_copies", 2) or 0))
        seed_program_mutation_replace_rate = float(getattr(self.config, "seed_program_mutation_replace_rate", 0.15) or 0.0)
        ga_top_n = max(1, int(getattr(self.config, "ga_top_n", 10) or 10))
        candidate_pool_size = max(ga_top_n, int(getattr(self.config, "ga_candidate_pool_size", ga_top_n * 2) or ga_top_n * 2))
        min_ic_threshold = float(getattr(self.config, "ga_min_ic_threshold", 0.0) or 0.0)
        max_similarity_threshold = float(getattr(self.config, "ga_max_similarity_threshold", 1.0) or 1.0)
        max_similarity_threshold = min(max(max_similarity_threshold, 0.0), 1.0)
        inverse_vol_weight_power = max(0.0, float(getattr(self.config, "ga_inverse_vol_weight_power", 0.0) or 0.0))
        downside_penalty = max(0.0, float(getattr(self.config, "ga_downside_penalty", 0.0) or 0.0))
        drawdown_penalty = max(0.0, float(getattr(self.config, "ga_drawdown_penalty", 0.0) or 0.0))
        volatility_penalty = max(0.0, float(getattr(self.config, "ga_volatility_penalty", 0.0) or 0.0))
        frequency_quota_targets = {
            "pure_daily": max(0, int(getattr(self.config, "ga_min_pure_daily", 0) or 0)),
            "pure_minute_derived": max(0, int(getattr(self.config, "ga_min_pure_minute_derived", 0) or 0)),
            "mixed_frequency": max(0, int(getattr(self.config, "ga_min_mixed_frequency", 0) or 0)),
        }
        enable_mixed_synthesis_fallback = bool(
            getattr(self.config, "ga_enable_mixed_synthesis_fallback", False)
        )
        frequency_quota_hard = bool(getattr(self.config, "ga_frequency_quota_hard", False))
        frequency_quota_enabled = bool(is_3d_mode and any(value > 0 for value in frequency_quota_targets.values()))
        selected_frequency_counts = self._empty_frequency_counter()

        dropped_duplicate = 0
        dropped_invalid_score = 0
        dropped_by_ic_threshold = 0
        dropped_by_similarity_threshold = 0

        gpu_enabled = False
        torch_device = None
        if use_gpu_requested and torch is not None and torch.cuda.is_available():
            try:
                torch_device = torch.device(f"cuda:{cuda_index}")
                _ = torch.empty(1, device=torch_device)
                gpu_enabled = True
                logger.info(f"GA scoring 使用 GPU: {torch_device}")
            except Exception as exc:
                logger.warning(f"GA scoring GPU 初始化失败，回退 CPU: {exc}")
                gpu_enabled = False
                torch_device = None
        elif use_gpu_requested:
            logger.info("GA scoring 未启用 GPU（torch 不可用或 CUDA 不可用），使用 CPU")

        # GPU 模式下避免 joblib 多进程竞争同一块显卡
        effective_workers = ga_workers
        if gpu_enabled and ga_workers > 1:
            logger.warning(
                f"检测到 GA GPU 模式，ga_workers 从 {ga_workers} 调整为 1 以避免多进程争抢显卡"
            )
            effective_workers = 1

        if inject_seed_programs and seed_programs:
            logger.info(
                "GP 初始种群注入 seed: %d 个原始表达式, 每个额外生成 %d 个轻微扰动副本",
                len(seed_programs),
                seed_program_mutation_copies,
            )

        if frequency_quota_enabled:
            min_required = sum(frequency_quota_targets.values())
            if self.config.num_factors < min_required:
                logger.warning(
                    "频率配额最小需求(%d)超过目标因子数(%d)，无法完全满足。",
                    min_required,
                    self.config.num_factors,
                )

        def _to_cross_section_matrix(values: np.ndarray) -> np.ndarray | None:
            array = np.asarray(values, dtype=float)
            if array.ndim == 1:
                if array.size % n_stocks != 0:
                    return None
                return array.reshape(-1, n_stocks)
            if array.ndim == 2:
                return array
            return None

        def _spearman_1d_np(a: np.ndarray, b: np.ndarray) -> float:
            try:
                ra = np.argsort(np.argsort(a))
                rb = np.argsort(np.argsort(b))
            except Exception:
                return float("nan")

            ra = ra.astype(float)
            rb = rb.astype(float)
            ra -= ra.mean()
            rb -= rb.mean()
            denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
            if denom <= 1e-12:
                return float("nan")
            return float(np.dot(ra, rb) / denom)

        def _mean_abs_spearman_np(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
            a_mat = _to_cross_section_matrix(flat_a)
            b_mat = _to_cross_section_matrix(flat_b)
            if a_mat is None or b_mat is None or a_mat.shape != b_mat.shape:
                return 0.0

            row_scores: List[float] = []
            for row_index in range(a_mat.shape[0]):
                ar = a_mat[row_index]
                br = b_mat[row_index]
                mask = np.isfinite(ar) & np.isfinite(br)
                if mask.sum() < 10:
                    continue

                arv = ar[mask]
                brv = br[mask]
                if np.nanstd(arv) <= 1e-12 or np.nanstd(brv) <= 1e-12:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = _spearman_1d_np(arv, brv)
                if not np.isnan(corr):
                    row_scores.append(corr)

            if not row_scores:
                return 0.0

            min_valid_ic_days = self._resolve_min_valid_ic_days(a_mat.shape[0])
            if len(row_scores) < min_valid_ic_days:
                return 0.0
            
            ic_array = np.array(row_scores)
            metric_type = getattr(self.config, "fitness_metric", "mean_abs_ic")
            return self._reduce_ic_values(ic_array, metric_type)

        def _mean_abs_spearman_gpu(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
            if not gpu_enabled or torch is None or torch_device is None:
                return _mean_abs_spearman_np(flat_a, flat_b)

            try:
                a_np = _to_cross_section_matrix(flat_a)
                b_np = _to_cross_section_matrix(flat_b)
                if a_np is None or b_np is None or a_np.shape != b_np.shape:
                    return 0.0
                with torch.no_grad():
                    a_mat = torch.as_tensor(a_np, dtype=torch.float32, device=torch_device)
                    b_mat = torch.as_tensor(b_np, dtype=torch.float32, device=torch_device)
                    row_scores: List = []

                    for row_index in range(a_mat.shape[0]):
                        ar = a_mat[row_index]
                        br = b_mat[row_index]
                        mask = torch.isfinite(ar) & torch.isfinite(br)
                        if int(mask.sum().item()) < 10:
                            continue

                        arv = ar[mask]
                        brv = br[mask]
                        if torch.std(arv, unbiased=False) <= 1e-12 or torch.std(brv, unbiased=False) <= 1e-12:
                            continue

                        ra = torch.argsort(torch.argsort(arv)).to(torch.float32)
                        rb = torch.argsort(torch.argsort(brv)).to(torch.float32)
                        ra = ra - torch.mean(ra)
                        rb = rb - torch.mean(rb)
                        denom = torch.sqrt(torch.sum(ra * ra) * torch.sum(rb * rb))
                        if float(denom.item()) <= 1e-12:
                            continue
                        corr = torch.sum(ra * rb) / denom
                        row_scores.append(corr)

                    if not row_scores:
                        return 0.0

                    min_valid_ic_days = self._resolve_min_valid_ic_days(a_mat.shape[0])
                    if len(row_scores) < min_valid_ic_days:
                        return 0.0

                    scores_tensor = torch.stack(row_scores)
                    metric_type = str(getattr(self.config, "fitness_metric", "mean_abs_ic") or "mean_abs_ic").strip().lower()

                    if metric_type == "icir":
                        ic_mean = torch.mean(scores_tensor)
                        ic_std = torch.std(scores_tensor, unbiased=True)
                        return float((torch.abs(ic_mean) / (ic_std + 1e-6)).item())
                    if metric_type == "mean_ic":
                        return float(torch.abs(torch.mean(scores_tensor)).item())
                    if metric_type == "signed_mean_ic":
                        return float(torch.mean(scores_tensor).item())
                    return float(torch.mean(torch.abs(scores_tensor)).item())
            except Exception as exc:
                logger.debug(f"GA GPU scoring 失败，回退 CPU: {exc}")
                return _mean_abs_spearman_np(flat_a, flat_b)

        def _mean_abs_spearman(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
            if gpu_enabled:
                return _mean_abs_spearman_gpu(flat_a, flat_b)
            return _mean_abs_spearman_np(flat_a, flat_b)

        def _compute_portfolio_risk_penalty(flat_true: np.ndarray, flat_pred: np.ndarray) -> float:
            if (downside_penalty + drawdown_penalty + volatility_penalty) <= 0.0:
                return 0.0

            try:
                true_mat = _to_cross_section_matrix(flat_true)
                pred_mat = _to_cross_section_matrix(flat_pred)
                if true_mat is None or pred_mat is None or true_mat.shape != pred_mat.shape:
                    return 0.0
                returns_path: List[float] = []
                selected_vol_path: List[float] = []

                for row_index in range(true_mat.shape[0]):
                    true_row = true_mat[row_index]
                    pred_row = pred_mat[row_index]
                    vol_row = volatility_matrix[row_index]
                    mask = np.isfinite(true_row) & np.isfinite(pred_row) & np.isfinite(vol_row)
                    if int(mask.sum()) < ga_top_n:
                        continue

                    true_valid = true_row[mask]
                    pred_valid = pred_row[mask]
                    vol_valid = np.clip(np.abs(vol_row[mask]), 1e-6, None)

                    rank_order = np.argsort(pred_valid)[::-1]
                    pool_size = min(candidate_pool_size, rank_order.size)
                    if pool_size <= 0:
                        continue

                    candidate_idx = rank_order[:pool_size]
                    candidate_pred = pred_valid[candidate_idx]
                    candidate_vol = vol_valid[candidate_idx]

                    if inverse_vol_weight_power > 0.0:
                        risk_adjusted = candidate_pred / np.power(candidate_vol, inverse_vol_weight_power)
                    else:
                        risk_adjusted = candidate_pred

                    choose_count = min(ga_top_n, risk_adjusted.size)
                    chosen_local_idx = np.argsort(risk_adjusted)[::-1][:choose_count]
                    chosen_idx = candidate_idx[chosen_local_idx]
                    chosen_returns = true_valid[chosen_idx]
                    chosen_vol = vol_valid[chosen_idx]

                    if inverse_vol_weight_power > 0.0:
                        raw_weights = 1.0 / np.power(chosen_vol, inverse_vol_weight_power)
                    else:
                        raw_weights = np.ones_like(chosen_vol)
                    raw_weights = np.clip(raw_weights, 1e-12, None)
                    weights = raw_weights / raw_weights.sum()

                    returns_path.append(float(np.dot(weights, chosen_returns)))
                    selected_vol_path.append(float(np.dot(weights, chosen_vol)))

                if not returns_path:
                    return 0.0

                returns_array = np.asarray(returns_path, dtype=float)
                downside_dev = float(np.sqrt(np.mean(np.square(np.minimum(returns_array, 0.0)))))
                curve = np.cumprod(1.0 + np.clip(returns_array, -0.95, None))
                running_peak = np.maximum.accumulate(curve)
                max_drawdown = float(np.abs(np.min(curve / running_peak - 1.0)))
                avg_selected_vol = float(np.mean(selected_vol_path)) if selected_vol_path else 0.0
                return (
                    downside_penalty * downside_dev
                    + drawdown_penalty * max_drawdown
                    + volatility_penalty * avg_selected_vol
                )
            except Exception:
                return 0.0

        def scoring(y_true, y_pred, _sample_weight) -> float:
            """自定义打分函数：在 IC/ICIR 基础上叠加组合下行风险与回撤惩罚。"""
            metric_type = str(getattr(self.config, "fitness_metric", "mean_abs_ic") or "mean_abs_ic").strip().lower()
            true_matrix = _to_cross_section_matrix(y_true)
            pred_matrix = _to_cross_section_matrix(y_pred)
            valid_matrix = _to_cross_section_matrix(valid_flat)
            if true_matrix is None or pred_matrix is None or true_matrix.shape != pred_matrix.shape:
                with np.errstate(divide="ignore", invalid="ignore"):
                    try:
                        corr = _spearman_1d_np(np.asarray(y_true), np.asarray(y_pred))
                    except Exception:
                        corr = np.nan
                    return float(abs(corr)) if not np.isnan(corr) else 0.0

            true_series = np.asarray(true_matrix, dtype=float).copy()
            pred_series = np.asarray(pred_matrix, dtype=float).copy()
            if valid_matrix is not None and valid_matrix.shape == true_series.shape:
                valid_mask_local = valid_matrix.astype(bool)
            else:
                valid_mask_local = np.isfinite(true_series)
            true_series[~valid_mask_local] = np.nan
            pred_series[~valid_mask_local] = np.nan

            mean_score = _mean_abs_spearman(true_series, pred_series)
            if metric_type != "signed_mean_ic":
                mean_score = max(0.0, mean_score)
            risk_penalty_value = _compute_portfolio_risk_penalty(true_series, pred_series)
            final_score = mean_score - risk_penalty_value
            if metric_type != "signed_mean_ic":
                final_score = max(0.0, final_score)

            # 若已选择集合非空，则对候选因子与已选择因子的相关度施加惩罚，鼓励多样性
            try:
                if selected_preds_list:
                    # 计算与每个已选因子的相关性并取平均作为总体相似度度量
                    corr_vals = []
                    for sp in selected_preds_list:
                        try:
                            corr_vals.append(_mean_abs_spearman(pred_series, sp))
                        except Exception:
                            continue

                    mean_abs_corr_with_selected = float(np.mean(corr_vals)) if corr_vals else 0.0
                    alpha = float(self.config.diversity_penalty_alpha if hasattr(self.config, 'diversity_penalty_alpha') else 0.3)
                    penalised = final_score - alpha * mean_abs_corr_with_selected
                    if metric_type != "signed_mean_ic":
                        final_score = max(0.0, penalised)
                    else:
                        final_score = penalised
                    return final_score
            except Exception:
                # 任何异常都退回到仅包含风险惩罚的得分
                pass

            return final_score

        logger.info("开始GA因子挖掘...")

        # 检查是否可以跳过重练（简单可靠的断点续传）
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        latest_path = output_dir / "generated_factors_latest.txt"
        
        # 如果文件存在且修改时间在 1 小时内，且文件不为空，则尝试直接读取
        # 【修复】：当指定了 cutoff_date (即在回测滚动过程中) 时，强制重新训练，不可复用缓存
        if cutoff_date is None and latest_path.exists() and latest_path.stat().st_size > 100:
            resume_cache_hours = max(0.0, float(getattr(self.config, "ga_resume_cache_hours", 1.0) or 0.0))
            mtime = latest_path.stat().st_mtime
            if (time.time() - mtime) < (resume_cache_hours * 3600):
                logger.info(f"检测到最近({resume_cache_hours:.2f}小时内)生成的因子文件: {latest_path}, 尝试直接加载跳过重练...")
                try:
                    loaded_exprs = []
                    for line in latest_path.read_text(encoding="utf-8").splitlines():
                        if "#" in line and "score=" in line:
                            parts = line.split("# score=")
                            expr_part = parts[0].split(":", 1)[1].strip()
                            score_part = float(parts[1].strip())
                            loaded_exprs.append(FactorExpression(expr_part, score_part))
                    if len(loaded_exprs) >= self.config.num_factors:
                        logger.info(f"成功恢复 {len(loaded_exprs)} 个因子，跳过 GA 训练阶段。")
                        return loaded_exprs
                except Exception as exc:
                    logger.warning(f"尝试恢复因子失败: {exc}, 将重新开始挖掘。")

        # 准备输出目录与迭代进度文件，确保每次迭代结果都被记录
        progress_path = output_dir / "generated_factors_progress.txt"
        # 写入表头（如果文件为空）
        if not progress_path.exists():
            progress_path.write_text("# iter\ttimestamp\tIC\texpression\n", encoding="utf-8")
        selected_progress_path = output_dir / "selected_factors_progress.txt"
        if not selected_progress_path.exists():
            selected_progress_path.write_text("# iter\ttimestamp\tIC\texpression\n", encoding="utf-8")
        frequency_summary_path = output_dir / "frequency_summary.txt"
        if not frequency_summary_path.exists():
            frequency_summary_path.write_text(
                "# iter\ttimestamp\tphase\tpure_daily\tpure_minute_derived\tmixed_frequency\tunknown\ttotal\n",
                encoding="utf-8",
            )
        default_max_iterations = max(
            self.config.num_factors,
            int(self.config.num_factors * max(1, int(getattr(self.config, "ga_iteration_multiplier", 2) or 1))),
        )
        configured_max_iterations = getattr(self.config, "ga_max_iterations", None)
        if configured_max_iterations is None:
            max_iterations = default_max_iterations
        else:
            max_iterations = max(1, int(configured_max_iterations))
            if max_iterations < self.config.num_factors:
                logger.warning(
                    "检测到 ga_max_iterations(%d) < num_factors(%d)，可能导致最终因子数不足。",
                    max_iterations,
                    self.config.num_factors,
                )
        factors_per_iteration = max(1, int(getattr(self.config, "ga_factors_per_iteration", 1) or 1))
        seed_base = int(getattr(self.config, "ga_random_seed_base", 42) or 42)
        model_feature_types = None
        if is_3d_mode:
            daily_count = max(0, int(getattr(self, "_training_daily_feature_count", 0) or 0))
            minute_count = max(0, int(getattr(self, "_training_minute_feature_count", 0) or 0))
            total_count = len(model_feature_names)
            if daily_count + minute_count != total_count:
                minute_count = max(0, total_count - daily_count)
            model_feature_types = [DAILY_MATRIX] * daily_count + [MINUTE_TENSOR] * minute_count
            daily_feature_name_set = set(model_feature_names[:daily_count])
        else:
            daily_feature_name_set = set(model_feature_names)
        for iteration in tqdm(range(max_iterations), desc="GA训练", unit="iter"):
            model = my_gplearn(
                list(self.function_set),
                scoring,
                pop_num=pop_num,
                gen_num=gen_num,
                random_state=seed_base + iteration,
                feature_names=model_feature_names,
                feature_types=model_feature_types,
                seed_programs=seed_programs if inject_seed_programs else None,
                seed_program_mutation_copies=seed_program_mutation_copies,
                seed_program_mutation_replace_rate=seed_program_mutation_replace_rate,
                typed_root_reducer_mode=is_3d_mode,
                p_crossover=float(self.config.ga_p_crossover),
                p_subtree_mutation=float(self.config.ga_p_subtree_mutation),
                p_hoist_mutation=float(self.config.ga_p_hoist_mutation),
                p_point_mutation=float(self.config.ga_p_point_mutation),
                verbose=0,
                n_jobs=effective_workers,
            )
            try:
                model.fit(x_array, y_array)
            except Exception as exc:
                logger.exception(f"模型训练失败（迭代 {iteration+1}）：{exc}")
                continue
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            round_added = 0
            round_candidates: List[_RoundCandidate] = []
            round_seen_expressions: Set[str] = set()
            round_generated_counts = self._empty_frequency_counter()
            round_selected_counts = self._empty_frequency_counter()
            for candidate in self._iter_round_candidates(model):
                expr = self._candidate_expression(candidate)
                if expr in round_seen_expressions:
                    dropped_duplicate += 1
                    logger.debug("  -> 当前轮次候选重复，跳过")
                    continue
                round_seen_expressions.add(expr)

                if any(item.expression == expr for item in expressions):
                    dropped_duplicate += 1
                    logger.debug("  -> 重复因子，跳过")
                    continue

                try:
                    y_pred = self._candidate_predict(model, candidate, x_array)
                    score = scoring(y_array, y_pred, None)
                except Exception as exc:
                    logger.warning(f"模型预测或评分失败（迭代 {iteration+1}, expr={expr}）：{exc}")
                    continue

                logger.info(f"迭代 {iteration+1}: IC={score:.4f} :: {expr}")
                try:
                    with progress_path.open("a", encoding="utf-8") as pf:
                        pf.write(f"{iteration+1}\t{ts}\t{score:.6f}\t{expr}\n")
                        pf.flush()
                except Exception as exc:
                    logger.warning(f"写入进度文件失败: {exc}")

                if np.isnan(score) or np.isinf(score):
                    dropped_invalid_score += 1
                    logger.warning(f"  -> 过滤异常因子 (IC={score})")
                    continue

                metric_type = str(getattr(self.config, "fitness_metric", "mean_abs_ic") or "mean_abs_ic").strip().lower()
                threshold_score = abs(score) if metric_type == "signed_mean_ic" else score

                if threshold_score < min_ic_threshold:
                    dropped_by_ic_threshold += 1
                    logger.debug(
                        "  -> 低于 IC 硬阈值，跳过 (score=%.6f < min=%.6f)",
                        threshold_score,
                        min_ic_threshold,
                    )
                    continue

                frequency_type = self._classify_expression_frequency(expr, daily_feature_name_set)
                round_generated_counts[frequency_type] = round_generated_counts.get(frequency_type, 0) + 1

                round_candidates.append(
                    _RoundCandidate(
                        expression=expr,
                        score=float(score),
                        prediction=np.asarray(y_pred, dtype=float),
                        frequency_type=frequency_type,
                    )
                )

            if (
                frequency_quota_enabled
                and enable_mixed_synthesis_fallback
                and frequency_quota_targets.get("mixed_frequency", 0) > 0
            ):
                has_mixed_candidate = any(item.frequency_type == "mixed_frequency" for item in round_candidates)
                if not has_mixed_candidate and "d_add" in self.function_set:
                    best_daily = next((item for item in round_candidates if item.frequency_type == "pure_daily"), None)
                    best_minute = next((item for item in round_candidates if item.frequency_type == "pure_minute_derived"), None)
                    if best_daily is not None and best_minute is not None:
                        synthetic_expr = f"d_add({best_daily.expression}, {best_minute.expression})"
                        exists_in_round = any(item.expression == synthetic_expr for item in round_candidates)
                        exists_in_selected = any(item.expression == synthetic_expr for item in expressions)
                        if not exists_in_round and not exists_in_selected:
                            try:
                                synthetic_pred = np.asarray(best_daily.prediction, dtype=float) + np.asarray(best_minute.prediction, dtype=float)
                                synthetic_score = float(scoring(y_array, synthetic_pred, None))
                                if np.isfinite(synthetic_score) and synthetic_score >= min_ic_threshold:
                                    round_candidates.append(
                                        _RoundCandidate(
                                            expression=synthetic_expr,
                                            score=synthetic_score,
                                            prediction=synthetic_pred,
                                            frequency_type="mixed_frequency",
                                        )
                                    )
                                    round_generated_counts["mixed_frequency"] = round_generated_counts.get("mixed_frequency", 0) + 1
                                    logger.info(
                                        "迭代 %d: 合成混频候选 IC=%.4f :: %s",
                                        iteration + 1,
                                        synthetic_score,
                                        synthetic_expr,
                                    )
                                    try:
                                        with progress_path.open("a", encoding="utf-8") as pf:
                                            pf.write(f"{iteration+1}\t{ts}\t{synthetic_score:.6f}\t{synthetic_expr}\n")
                                            pf.flush()
                                    except Exception as exc:
                                        logger.warning(f"写入合成混频候选失败: {exc}")
                            except Exception as exc:
                                logger.debug(f"合成混频候选失败: {exc}")

            # 严格按最终 score 排序后取 TopK，而不是按遍历顺序先到先得。
            round_candidates.sort(key=lambda item: item.score, reverse=True)
            round_target_additions = min(
                factors_per_iteration,
                max(0, int(self.config.num_factors) - len(expressions)),
            )
            attempted_indices: Set[int] = set()

            def _try_select(index: int) -> bool:
                nonlocal round_added, dropped_by_similarity_threshold
                if index in attempted_indices:
                    return False
                attempted_indices.add(index)

                selected = round_candidates[index]
                if selected_preds_list and max_similarity_threshold < 1.0:
                    pred_series = np.asarray(selected.prediction, dtype=float).copy()
                    pred_series[~valid_flat] = np.nan
                    corr_vals: List[float] = []
                    for selected_pred in selected_preds_list:
                        try:
                            corr_vals.append(_mean_abs_spearman(pred_series, selected_pred))
                        except Exception:
                            continue

                    max_abs_corr_with_selected = float(np.max(corr_vals)) if corr_vals else 0.0
                    if max_abs_corr_with_selected > max_similarity_threshold:
                        dropped_by_similarity_threshold += 1
                        logger.debug(
                            "  -> 超过相似度硬阈值，跳过 (max_corr=%.6f > max=%.6f)",
                            max_abs_corr_with_selected,
                            max_similarity_threshold,
                        )
                        return False

                expressions.append(FactorExpression(selected.expression, selected.score))
                try:
                    selected_preds_list.append(selected.prediction.copy())
                except Exception:
                    selected_preds_list.append(np.array(selected.prediction))
                round_added += 1
                round_selected_counts[selected.frequency_type] = round_selected_counts.get(selected.frequency_type, 0) + 1
                selected_frequency_counts[selected.frequency_type] = selected_frequency_counts.get(selected.frequency_type, 0) + 1
                logger.info(f"  -> 添加新因子 (累计{len(expressions)}个)")
                try:
                    with selected_progress_path.open("a", encoding="utf-8") as spf:
                        spf.write(f"{iteration+1}\t{ts}\t{selected.score:.6f}\t{selected.expression}\n")
                        spf.flush()
                except Exception as exc:
                    logger.warning(f"写入最终入选进度失败: {exc}")
                return True

            if frequency_quota_enabled and round_target_additions > 0:
                for category in ("pure_daily", "pure_minute_derived", "mixed_frequency"):
                    if round_added >= round_target_additions:
                        break
                    missing = max(0, frequency_quota_targets.get(category, 0) - selected_frequency_counts.get(category, 0))
                    if missing <= 0:
                        continue
                    for idx, candidate in enumerate(round_candidates):
                        if round_added >= round_target_additions or missing <= 0:
                            break
                        if candidate.frequency_type != category or idx in attempted_indices:
                            continue
                        if _try_select(idx):
                            missing -= 1

            for idx, _ in enumerate(round_candidates):
                if round_added >= round_target_additions:
                    break
                if idx in attempted_indices:
                    continue
                _try_select(idx)

            try:
                generated_total = sum(round_generated_counts.values())
                selected_total = sum(round_selected_counts.values())
                with frequency_summary_path.open("a", encoding="utf-8") as fs:
                    fs.write(
                        f"{iteration+1}\t{ts}\tgenerated\t"
                        f"{round_generated_counts['pure_daily']}\t{round_generated_counts['pure_minute_derived']}\t"
                        f"{round_generated_counts['mixed_frequency']}\t{round_generated_counts['unknown']}\t{generated_total}\n"
                    )
                    fs.write(
                        f"{iteration+1}\t{ts}\tselected\t"
                        f"{round_selected_counts['pure_daily']}\t{round_selected_counts['pure_minute_derived']}\t"
                        f"{round_selected_counts['mixed_frequency']}\t{round_selected_counts['unknown']}\t{selected_total}\n"
                    )
                    fs.flush()
            except Exception as exc:
                logger.warning(f"写入频率统计失败: {exc}")

            logger.info(
                "第 %d/%d 轮，本轮新增 %d 个，累计 %d/%d 个因子",
                iteration + 1,
                max_iterations,
                round_added,
                len(expressions),
                self.config.num_factors,
            )
            logger.info(
                "频率统计(本轮入选): daily=%d, minute_derived=%d, mixed=%d, unknown=%d",
                round_selected_counts.get("pure_daily", 0),
                round_selected_counts.get("pure_minute_derived", 0),
                round_selected_counts.get("mixed_frequency", 0),
                round_selected_counts.get("unknown", 0),
            )
            if frequency_quota_enabled:
                logger.info(
                    "频率配额剩余: daily=%d, minute_derived=%d, mixed=%d",
                    max(0, frequency_quota_targets.get("pure_daily", 0) - selected_frequency_counts.get("pure_daily", 0)),
                    max(0, frequency_quota_targets.get("pure_minute_derived", 0) - selected_frequency_counts.get("pure_minute_derived", 0)),
                    max(0, frequency_quota_targets.get("mixed_frequency", 0) - selected_frequency_counts.get("mixed_frequency", 0)),
                )
                
            if len(expressions) >= self.config.num_factors:
                break

        if len(expressions) < self.config.num_factors:
            logger.warning(
                "GA 达到最大迭代仍未凑齐目标因子数：当前 %d / 目标 %d。"
                "硬筛配置: ga_min_ic_threshold=%.6f, ga_max_similarity_threshold=%.6f",
                len(expressions),
                self.config.num_factors,
                min_ic_threshold,
                max_similarity_threshold,
            )

        if frequency_quota_enabled:
            quota_unmet = {
                category: max(0, frequency_quota_targets.get(category, 0) - selected_frequency_counts.get(category, 0))
                for category in ("pure_daily", "pure_minute_derived", "mixed_frequency")
            }
            if any(value > 0 for value in quota_unmet.values()):
                message = (
                    "频率配额未满足: "
                    f"daily缺{quota_unmet['pure_daily']}, "
                    f"minute_derived缺{quota_unmet['pure_minute_derived']}, "
                    f"mixed缺{quota_unmet['mixed_frequency']}"
                )
                if frequency_quota_hard:
                    raise RuntimeError(message)
                logger.warning(message)

        logger.info(
            "GA 候选过滤统计: 重复=%d, 异常得分=%d, IC阈值=%d, 相似度阈值=%d",
            dropped_duplicate,
            dropped_invalid_score,
            dropped_by_ic_threshold,
            dropped_by_similarity_threshold,
        )

        expressions.sort(key=lambda item: item.score, reverse=True)
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"generated_factors_{timestamp}.txt"

        lines = ["# Generated factor expressions\n", f"# Created at: {timestamp}\n"]
        # 将结果追加为类似 FactorXX: 表达式 # score= 的格式，方便人工筛选
        for index, item in enumerate(expressions, start=1):
            lines.append(f"Factor{index:02d}: {item.expression}  # score={item.score:.4f}\n")

        with output_path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines)

        latest_path = output_dir / "generated_factors_latest.txt"
        latest_path.write_text("".join(lines), encoding="utf-8")

        return expressions


def load_ga_config() -> GAFactorConfig:
    cfg = get_config_value("ga_pipeline")
    raw_function_set = cfg.get("function_set")
    parsed_function_set: Tuple[str, ...] | None = None
    if isinstance(raw_function_set, (list, tuple)):
        parsed_function_set = tuple(str(item).strip() for item in raw_function_set if str(item).strip()) or None
    raw_daily_field_names = cfg.get("daily_field_names")
    parsed_daily_field_names: Tuple[str, ...] | None = None
    if isinstance(raw_daily_field_names, (list, tuple)):
        parsed_daily_field_names = tuple(str(item).strip() for item in raw_daily_field_names if str(item).strip()) or None
    raw_minute_field_names = cfg.get("minute_field_names")
    parsed_minute_field_names: Tuple[str, ...] | None = None
    if isinstance(raw_minute_field_names, (list, tuple)):
        parsed_minute_field_names = tuple(str(item).strip() for item in raw_minute_field_names if str(item).strip()) or None

    return GAFactorConfig(
        num_factors=cfg.get("num_factors", 5),
        train_months=cfg.get("train_months", 12),
        predict_days=cfg.get("predict_days", 5),
        output_directory=cfg.get("output_directory", "/home/intership/guojh/GA和GRU/factors"),
        price_file=cfg.get("price_file"),
        factor_file=cfg.get("factor_file"),
        max_factors=cfg.get("max_factors", 25)
        ,daily_field_selection_mode=str(cfg.get("daily_field_selection_mode", "auto")).lower()
        ,daily_field_names=parsed_daily_field_names
        ,diversity_penalty_alpha=cfg.get("diversity_penalty_alpha", 0.3)
        ,seed_expression_file=cfg.get("seed_expression_file")
        ,max_seed_expressions=cfg.get("max_seed_expressions", 5)
        ,inject_seed_programs=cfg.get("inject_seed_programs", True)
        ,seed_program_mutation_copies=cfg.get("seed_program_mutation_copies", 2)
        ,seed_program_mutation_replace_rate=cfg.get("seed_program_mutation_replace_rate", 0.15)
        ,fitness_metric=cfg.get("fitness_metric", "mean_abs_ic")
        ,ga_min_ic_threshold=cfg.get("ga_min_ic_threshold", 0.0)
        ,ga_min_valid_ic_days=cfg.get("ga_min_valid_ic_days", 0)
        ,ga_max_similarity_threshold=cfg.get("ga_max_similarity_threshold", 1.0)
        ,ga_workers=cfg.get("ga_workers", 50)
        ,ga_pop_num=cfg.get("ga_pop_num", 500)
        ,ga_gen_num=cfg.get("ga_gen_num", 6)
        ,ga_p_crossover=cfg.get("ga_p_crossover", 0.8)
        ,ga_p_subtree_mutation=cfg.get("ga_p_subtree_mutation", 0.05)
        ,ga_p_hoist_mutation=cfg.get("ga_p_hoist_mutation", 0.05)
        ,ga_p_point_mutation=cfg.get("ga_p_point_mutation", 0.05)
        ,ga_use_gpu=cfg.get("ga_use_gpu", True)
        ,ga_cuda_device=cfg.get("ga_cuda_device", 0)
        ,ga_top_n=cfg.get("ga_top_n", 10)
        ,ga_candidate_pool_size=cfg.get("ga_candidate_pool_size", 20)
        ,ga_inverse_vol_weight_power=cfg.get("ga_inverse_vol_weight_power", 0.0)
        ,ga_downside_penalty=cfg.get("ga_downside_penalty", 0.12)
        ,ga_drawdown_penalty=cfg.get("ga_drawdown_penalty", 0.18)
        ,ga_volatility_penalty=cfg.get("ga_volatility_penalty", 0.05)
        ,ga_random_seed_base=cfg.get("ga_random_seed_base", 42)
        ,ga_iteration_multiplier=cfg.get("ga_iteration_multiplier", 2)
        ,ga_max_iterations=cfg.get("ga_max_iterations")
        ,ga_factors_per_iteration=cfg.get("ga_factors_per_iteration", 1)
        ,ga_round_candidate_multiplier=cfg.get("ga_round_candidate_multiplier", 3)
        ,ga_min_pure_daily=cfg.get("ga_min_pure_daily", 0)
        ,ga_min_pure_minute_derived=cfg.get("ga_min_pure_minute_derived", 0)
        ,ga_min_mixed_frequency=cfg.get("ga_min_mixed_frequency", 0)
        ,ga_enable_mixed_synthesis_fallback=cfg.get("ga_enable_mixed_synthesis_fallback", False)
        ,ga_frequency_quota_hard=cfg.get("ga_frequency_quota_hard", False)
        ,ga_resume_cache_hours=cfg.get("ga_resume_cache_hours", 1.0)
        ,ga_volatility_window_min=cfg.get("ga_volatility_window_min", 5)
        ,ga_volatility_window_max=cfg.get("ga_volatility_window_max", 20)
        ,target_mode=str(cfg.get("target_mode", "legacy_price_return"))
        ,target_parquet_root=cfg.get("target_parquet_root")
        ,target_train_field=str(cfg.get("target_train_field", "neu_all_vwap_5d"))
        ,signal_to_entry_days=int(cfg.get("signal_to_entry_days", 1))
        ,target_horizon_days=int(cfg.get("target_horizon_days", 5))
        ,function_library_profile=str(cfg.get("function_library_profile", "balanced")).lower()
        ,function_set=parsed_function_set
        ,enable_time_functions=bool(cfg.get("enable_time_functions", True))
        ,enable_wq_time_functions=bool(cfg.get("enable_wq_time_functions", False))
        ,enable_logic_functions=bool(cfg.get("enable_logic_functions", False))
        ,enable_trig_functions=bool(cfg.get("enable_trig_functions", False))
        ,enable_3d_mining=bool(cfg.get("enable_3d_mining", False))
        ,minute_cache_root=cfg.get("minute_cache_root")
        ,minute_field_names=parsed_minute_field_names
        ,minute_field_prefix=str(cfg.get("minute_field_prefix", "min_"))
        ,ga_3d_use_memmap=bool(cfg.get("ga_3d_use_memmap", True))
        ,ga_3d_memmap_dir=cfg.get("ga_3d_memmap_dir")
        ,ga_3d_dtype=str(cfg.get("ga_3d_dtype", "float32"))
        ,ga_3d_loader_workers=int(cfg.get("ga_3d_loader_workers", 1))
        ,ga_3d_loader_prefetch=int(cfg.get("ga_3d_loader_prefetch", 2))
        ,ga_3d_normalize_workers=int(cfg.get("ga_3d_normalize_workers", 1))
        ,ga_3d_shift_daily_features=bool(cfg.get("ga_3d_shift_daily_features", False))
    )
