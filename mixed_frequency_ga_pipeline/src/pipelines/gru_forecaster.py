"""基于 GRU 的因子序列建模与预测流程。

功能拆分为两部分：1）训练器：生成并保存 GRU 模型参数；2）预测器：加载既有模型，
在最新数据上输出股票推荐名单。这两个流程可以独立执行。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pickle
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
try:
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover - optional dependency for lr model_type
    Ridge = None

from gplearn.functions import _function_map
from gplearn.typed_functions import TYPED_FUNCTION_MAP

from src.utils.config_loader import get_config_value
from src.utils.data_io import (
    load_minute_supported_stock_columns,
    load_price_and_factors,
    restrict_price_factors_to_columns,
)
from src.utils.minute_tensor_cache import get_cache_file_path, load_minute_day_cache, load_minute_day_cache_axes
from src.utils.vwap_target_loader import build_target_matrix, load_vwap_target_long


@dataclass
class GRUConfig:
    """GRU 训练与推理所需的超参数与输入输出路径设定。"""
    sequence_length: int
    predict_days: int
    hidden_units: int
    batch_size: int
    epochs: int
    validation_ratio: float
    factor_expression_file: str
    price_file: str
    factor_file: str
    model_output_dir: str
    log_output_dir: str
    backtest_price_file: str | None = None
    backtest_factor_file: str | None = None
    recommendation_output_dir: str | None = None
    top_n: int | None = None
    train_ratio: float | None = None
    # 添加新参数以控制模型复杂度
    num_layers: int = 3
    dropout_rate: float = 0.3
    use_bidirectional: bool = True
    # 指定使用哪一张 GPU（0 表示第一张卡）
    cuda_device: int = 0
    # 是否对因子按截面（每日期）进行 z-score 标准化
    normalize_factors: bool = True
    # 是否对目标（future returns）按截面进行 z-score 标准化
    normalize_target: bool = True
    uncertainty_samples: int = 10
    # 目标标签来源: legacy_price_return 或 external_* (例如 external_vwap_5d/external_raw1d)
    target_mode: str = "legacy_price_return"
    # 外部目标 parquet 根路径（支持单文件或目录多分片）
    target_parquet_root: str | None = None
    # 回测标签 parquet 根路径（可选；未配置时回退到 target_parquet_root）
    target_backtest_parquet_root: str | None = None
    # 训练标签字段（外部目标模式下使用）
    target_train_field: str = "neu_all_vwap_5d"
    # 回测标签字段（外部目标模式下使用）
    target_backtest_field: str = "raw_vwap_5d"
    # 信号到入场的交易日偏移（1 表示 T 盘后信号，T+1 09:35 入场）
    signal_to_entry_days: int = 1
    # 目标持有期天数（仅用于元数据与校验，不参与外部标签计算）
    target_horizon_days: int = 5
    # 回测收益口径: daily_nextday 或 rebalance_5d
    backtest_return_mode: str = "daily_nextday"
    # 是否启用 3D-native 表达式执行
    enable_3d_mining: bool = False
    # 分钟 cache 根目录
    minute_cache_root: str | None = None
    # 分钟字段列表
    minute_field_names: Tuple[str, ...] | None = None
    # 分钟字段名前缀
    minute_field_prefix: str = "min_"
    # 3D 计算 dtype
    ga_3d_dtype: str = "float32"
    # 下游模型类型: gru 或 lr
    model_type: str = "gru"
    # LR 超参数
    lr_alpha: float = 1.0
    lr_fit_intercept: bool = True
    lr_max_iter: int = 2000
    # 训练随机性控制
    random_seed: int = 42
    deterministic_training: bool = True
    # 仅使用最近 N 个月样本训练 GRU；0/None 表示使用全部历史样本
    fit_window_months: int | None = None


class GRUFactorForecaster:
    """提供 GRU 模型的训练与基于最新数据的股票推荐能力。"""

    def __init__(self, config: GRUConfig) -> None:
        self.config = config
        self.model: Any = None
        self.seq_len: int | None = None
        self.feature_dim: int | None = None
        self.panel_mode: bool = True
        self._external_target_cache: dict[str, pd.DataFrame] = {}
        self._minute_availability_mask: pd.DataFrame | None = None
        self._latest_sample_valid_mask: pd.DataFrame | None = None

    def _model_type(self) -> str:
        return str(getattr(self.config, "model_type", "gru") or "gru").strip().lower()

    def _configure_reproducibility(self) -> torch.Generator:
        seed = int(getattr(self.config, "random_seed", 42) or 42)
        deterministic = bool(getattr(self.config, "deterministic_training", True))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        else:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(False)
                except Exception:
                    pass

        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _function_map(self) -> Dict[str, Any]:
        return TYPED_FUNCTION_MAP if bool(getattr(self.config, "enable_3d_mining", False)) else _function_map

    def _resolve_train_ratio(self) -> float:
        raw_train_ratio = getattr(self.config, "train_ratio", None)
        if raw_train_ratio is not None:
            ratio = float(raw_train_ratio)
        else:
            ratio = 1.0 - float(getattr(self.config, "validation_ratio", 0.2) or 0.2)
        return min(max(ratio, 0.01), 0.99)

    def _restrict_fit_window(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_dates: pd.Index | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex | None]:
        if sample_dates is None:
            return X, y, None

        fit_window_months = int(getattr(self.config, "fit_window_months", 0) or 0)
        sample_index = pd.DatetimeIndex(pd.to_datetime(sample_dates))
        if fit_window_months <= 0 or len(sample_index) == 0:
            return X, y, sample_index

        latest_date = pd.Timestamp(sample_index[-1])
        window_start = latest_date - pd.DateOffset(months=fit_window_months)
        start_idx = int(sample_index.searchsorted(window_start, side="left"))
        if start_idx >= len(sample_index) - 1:
            start_idx = max(0, len(sample_index) - 2)

        restricted_X = X[start_idx:]
        restricted_y = y[start_idx:]
        restricted_dates = sample_index[start_idx:]
        print(
            "GRU fit window restricted -> "
            f"months={fit_window_months}, samples={len(restricted_dates)}, "
            f"dates={restricted_dates[0].date()} -> {restricted_dates[-1].date()}"
        )
        return restricted_X, restricted_y, restricted_dates

    def _minute_terminal_name(self, field_name: str) -> str:
        prefix = str(getattr(self.config, "minute_field_prefix", "min_") or "min_")
        return f"{prefix}{field_name}"

    def _minute_field_names(self) -> List[str]:
        raw_fields = getattr(self.config, "minute_field_names", None)
        if not raw_fields:
            return []
        return [str(name).strip() for name in raw_fields if str(name).strip()]

    def _build_minute_data_sources(
        self,
        price_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        if not bool(getattr(self.config, "enable_3d_mining", False)):
            return {}
        if not self.config.minute_cache_root:
            raise ValueError("enable_3d_mining=True 时必须配置 minute_cache_root")

        minute_field_names = self._minute_field_names()
        if not minute_field_names:
            raise ValueError("enable_3d_mining=True 时必须配置 minute_field_names")

        cache_root = Path(self.config.minute_cache_root)
        trade_dates = [pd.Timestamp(item) for item in price_df.index]
        stock_codes = [str(item) for item in price_df.columns]
        stock_to_pos = {code: idx for idx, code in enumerate(stock_codes)}
        dtype = np.dtype(str(getattr(self.config, "ga_3d_dtype", "float32") or "float32"))

        available_dates: List[pd.Timestamp] = []
        reference_minutes: List[str] | None = None
        for trade_date in trade_dates:
            cache_file = get_cache_file_path(cache_root, trade_date)
            if not cache_file.exists():
                continue
            try:
                cache_axes = load_minute_day_cache_axes(cache_root, trade_date)
            except Exception:
                continue
            minute_labels = [str(item) for item in cache_axes["minutes"].tolist()]
            cache_fields = {str(item) for item in cache_axes["fields"].tolist()}
            if any(field not in cache_fields for field in minute_field_names):
                continue
            if reference_minutes is None:
                reference_minutes = minute_labels
            elif minute_labels != reference_minutes:
                continue
            available_dates.append(trade_date)

        if reference_minutes is None:
            raise ValueError("未找到可用的分钟 cache，无法评估 3D 表达式")

        n_dates = len(price_df.index)
        n_stocks = len(stock_codes)
        n_minutes = len(reference_minutes)
        availability = np.zeros((n_dates, n_stocks), dtype=bool)
        minute_sources = {
            self._minute_terminal_name(field_name): np.full((n_dates, n_stocks, n_minutes), np.nan, dtype=dtype)
            for field_name in minute_field_names
        }

        date_to_pos = {pd.Timestamp(item): idx for idx, item in enumerate(price_df.index)}
        for trade_date in available_dates:
            date_pos = date_to_pos[pd.Timestamp(trade_date)]
            cache_payload = load_minute_day_cache(cache_root, trade_date, mmap_mode="r")
            cache_stocks = [str(item) for item in cache_payload["stocks"].tolist()]
            cache_fields = [str(item) for item in cache_payload["fields"].tolist()]
            cache_data = np.asarray(cache_payload["data"], dtype=dtype)
            cache_data = np.where(np.isfinite(cache_data), cache_data, np.nan)
            cache_field_pos = {name: idx for idx, name in enumerate(cache_fields)}
            source_pos_map = {code: idx for idx, code in enumerate(cache_stocks)}
            intersect_codes = [code for code in stock_codes if code in source_pos_map]
            if not intersect_codes:
                continue
            source_rows = np.array([source_pos_map[code] for code in intersect_codes], dtype=np.int64)
            target_rows = np.array([stock_to_pos[code] for code in intersect_codes], dtype=np.int64)
            availability[date_pos, target_rows] = True
            for field_name in minute_field_names:
                minute_sources[self._minute_terminal_name(field_name)][date_pos, target_rows, :] = cache_data[
                    source_rows, :, cache_field_pos[field_name]
                ]

        self._minute_availability_mask = pd.DataFrame(
            availability,
            index=price_df.index,
            columns=price_df.columns,
        )
        for key, value in minute_sources.items():
            minute_sources[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return minute_sources

    def _build_daily_availability_mask(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        close_frame = factor_dict.get("close", price_df).reindex(index=price_df.index, columns=price_df.columns)
        close_frame = close_frame.replace([np.inf, -np.inf], np.nan)
        return close_frame.notna() & close_frame.ne(0.0)

    def _build_feature_availability_mask(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        daily_valid_mask = self._build_daily_availability_mask(price_df, factor_dict)
        if bool(getattr(self.config, "enable_3d_mining", False)):
            minute_valid_mask = self._minute_availability_mask
            if minute_valid_mask is None:
                minute_valid_mask = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        else:
            minute_valid_mask = pd.DataFrame(True, index=price_df.index, columns=price_df.columns)
        minute_valid_mask = minute_valid_mask.reindex(index=price_df.index, columns=price_df.columns).fillna(False)
        return daily_valid_mask & minute_valid_mask

    def _build_expression_data_sources(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        data_sources: Dict[str, Any] = {name: frame.reindex(index=price_df.index, columns=price_df.columns) for name, frame in factor_dict.items()}
        if "close" not in data_sources:
            data_sources["close"] = price_df
        if bool(getattr(self.config, "enable_3d_mining", False)):
            data_sources.update(self._build_minute_data_sources(price_df))
        return data_sources

    def _load_external_target_matrix(self, field_name: str) -> pd.DataFrame:
        """Load and cache one external target field as date x stock matrix."""
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

    def load_data(
        self,
        price_file: str | None = None,
        factor_file: str | None = None,
    ) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """读取价格、因子数据，允许通过参数覆盖默认配置。"""
        price_path = price_file or self.config.price_file
        factor_path = factor_file or self.config.factor_file
        if not price_path or not factor_path:
            raise ValueError("GRU pipeline requires price and factor file paths in configuration")
        price_df, factor_dict = load_price_and_factors(price_path, factor_path)
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

    def load_factor_expressions(self) -> List[str]:
        """从 GA 的表达式文件提取可执行的符号因子。"""
        path = Path(self.config.factor_expression_file)
        if not path.exists():
            raise FileNotFoundError(path)
        expressions: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and ":" in line:
                expr = line.split(":", 1)[1].strip()
                if "#" in expr:
                    expr = expr.split("#", 1)[0].strip()
                if expr:
                    expressions.append(expr)
        return expressions

    # --- expression evaluation helpers -------------------------------------------------

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
        """递归解析符号表达式为 (节点类型, 内容) 的树结构。"""
        expr = expression.strip()
        if not expr:
            raise ValueError("空的因子表达式")
        
        # 检查是否为数值常量
        try:
            float(expr)
            return ("const", expr)
        except ValueError:
            pass
            
        if "(" not in expr:
            return ("var", expr)
        if not expr.endswith(")"):
            raise ValueError(f"表达式格式错误: {expression}")
        func_name_end = expr.index("(")
        func_name = expr[:func_name_end].strip()
        inner = expr[func_name_end + 1 : -1]
        arguments = [self._parse_expression(item) for item in self._split_arguments(inner)]
        return ("func", func_name, arguments)

    def _evaluate_node(
        self,
        node,
        data_sources: Dict[str, Any],
    ) -> Tuple[np.ndarray | float, pd.Index | None, pd.Index | None]:
        """根据解析树节点计算对应的矩阵值。"""
        kind = node[0]
        if kind == "var":
            name = node[1]
            if name not in data_sources:
                raise KeyError(name)
            value = data_sources[name]
            if isinstance(value, pd.DataFrame):
                return value.values, value.index, value.columns
            return np.asarray(value), None, None
            
        # 处理数值常量
        if kind == "const":
            value = float(node[1])
            return value, None, None

        func_name = node[1]
        children = node[2]
        function_map = self._function_map()
        if func_name not in function_map:
            raise KeyError(func_name)
        evaluated = [self._evaluate_node(child, data_sources) for child in children]
        arrays = [item[0] for item in evaluated]
        # Broadcast scalar constants to the shape of sibling array inputs so
        # typed binary ops (which require same-shape tensors) can evaluate
        # expressions like min(x, 0.362) safely.
        array_shape = next(
            (
                np.asarray(value).shape
                for value in arrays
                if not np.isscalar(value) and np.asarray(value).ndim > 0
            ),
            None,
        )
        if array_shape is not None:
            arrays = [
                np.full(array_shape, float(value), dtype=np.float64)
                if np.isscalar(value)
                else np.asarray(value)
                for value in arrays
            ]
        result = function_map[func_name](*arrays)
        index = next((item[1] for item in evaluated if item[1] is not None), None)
        columns = next((item[2] for item in evaluated if item[2] is not None), None)
        # 确保结果不会出现无穷大或NaN值
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(result), index, columns

    def _evaluate_expression(
        self,
        expression: str,
        data_sources: Dict[str, Any],
    ) -> pd.DataFrame:
        """对完整表达式求值，返回与行情对齐的数据框。"""
        node = self._parse_expression(expression)
        values, index, columns = self._evaluate_node(node, data_sources)
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"表达式最终结果不是二维矩阵: {expression}, ndim={values.ndim}")
        if index is None or columns is None:
            sample_frame = next(value for value in data_sources.values() if isinstance(value, pd.DataFrame))
            index = sample_frame.index
            columns = sample_frame.columns
        return pd.DataFrame(values, index=index, columns=columns)

    def create_dataset(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        expressions: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """生成训练张量、目标收益与目标日期索引。"""
        self.panel_mode = True
        factors = self._build_factor_tensor(price_df, factor_dict, expressions).astype(np.float32, copy=False)

        seq_len = int(self.config.sequence_length)
        signal_to_entry_days = max(0, int(getattr(self.config, "signal_to_entry_days", 1) or 0))
        target_mode = str(getattr(self.config, "target_mode", "legacy_price_return")).lower()
        feature_valid_mask = self._build_feature_availability_mask(price_df, factor_dict)

        external_target_enabled = target_mode != "legacy_price_return"

        if external_target_enabled:
            max_start = len(price_df) - seq_len - signal_to_entry_days + 1
            if max_start <= 0:
                raise ValueError("Dataset is too small for sequence_length and signal_to_entry_days")

            target_matrix = self._load_external_target_matrix(self.config.target_train_field)
            target_matrix = target_matrix.reindex(columns=price_df.columns)

            n_stocks = price_df.shape[1]
            n_features = factors.shape[2]
            X = np.empty((max_start, n_stocks, seq_len, n_features), dtype=np.float32)
            target_values = np.full((max_start, n_stocks), np.nan, dtype=np.float32)
            target_dates: List[pd.Timestamp] = []

            for start in range(max_start):
                end = start + seq_len
                entry_pos = end - 1 + signal_to_entry_days
                entry_date = pd.Timestamp(price_df.index[entry_pos])
                y_row = target_matrix.reindex([entry_date], columns=price_df.columns).iloc[0]
                X[start] = np.transpose(factors[start:end], (1, 0, 2))
                seq_valid = feature_valid_mask.iloc[start:end].all(axis=0)
                valid = seq_valid & y_row.notna()
                row_values = y_row.to_numpy(dtype=np.float32, copy=True)
                row_values[~valid.to_numpy()] = np.nan
                target_values[start] = row_values
                target_dates.append(entry_date)

            if not np.isfinite(target_values).any():
                raise ValueError("No aligned panel samples found in external target mode; check parquet/date coverage")

            target_frame = pd.DataFrame(target_values, index=pd.DatetimeIndex(target_dates), columns=price_df.columns)
        else:
            targets = price_df.pct_change(self.config.predict_days).shift(-self.config.predict_days)
            target_frame = targets.iloc[self.config.sequence_length : -self.config.predict_days]
            target_frame = target_frame.replace([np.inf, -np.inf], np.nan)

            max_start = len(price_df) - self.config.sequence_length - self.config.predict_days
            if max_start <= 0:
                raise ValueError("Dataset is too small for sequence_length and predict_days")

            n_stocks = price_df.shape[1]
            n_features = factors.shape[2]
            X = np.empty((max_start, n_stocks, seq_len, n_features), dtype=np.float32)
            panel_targets = np.full((max_start, n_stocks), np.nan, dtype=np.float32)
            target_dates: List[pd.Timestamp] = []
            for start in range(max_start):
                end = start + self.config.sequence_length
                X[start] = np.transpose(factors[start:end], (1, 0, 2))
                y_row = target_frame.iloc[start]
                seq_valid = feature_valid_mask.iloc[start:end].all(axis=0)
                valid = seq_valid & y_row.notna()
                row_values = y_row.to_numpy(dtype=np.float32, copy=True)
                row_values[~valid.to_numpy()] = np.nan
                panel_targets[start] = row_values
                target_dates.append(pd.Timestamp(target_frame.index[start]))
            target_frame = pd.DataFrame(panel_targets, index=pd.DatetimeIndex(target_dates), columns=price_df.columns)

        # 可选：对目标按截面做 z-score 标准化，避免绝对收益尺度差异影响模型学习
        if getattr(self.config, 'normalize_target', False):
            # 使用 pandas 的广播按行（日期）做标准化
            row_mean = target_frame.mean(axis=1)
            row_std = target_frame.std(axis=1).replace(0, 1.0)
            target_frame = target_frame.sub(row_mean, axis=0).div(row_std, axis=0)
        target_values = target_frame.values
        target_dates = target_frame.index
        self._latest_sample_valid_mask = target_frame.notna()

        if X.shape[0] != len(target_dates):
            raise ValueError(
                f"Feature/target sample mismatch: X={X.shape[0]}, y={len(target_dates)}"
            )
        return X, target_values, target_dates

    def build_model(
        self,
        input_size: int,
        output_size: int,
        num_layers: int | None = None,
        dropout_rate: float | None = None,
        use_bidirectional: bool | None = None,
    ) -> nn.Module:
        """构建包含多层双向 GRU 的预测模型。

        参数 num_layers/dropout_rate/use_bidirectional 可选地覆盖配置中的值，
        以便在训练或加载时动态调整模型结构。
        """
        # 使用传入参数或回退到配置值
        num_layers = int(num_layers if num_layers is not None else self.config.num_layers)
        dropout_rate = float(dropout_rate if dropout_rate is not None else self.config.dropout_rate)
        use_bidirectional = bool(use_bidirectional if use_bidirectional is not None else self.config.use_bidirectional)
        class GRUNet(nn.Module):
            def __init__(self, in_features: int, hidden: int, out_features: int, 
                         num_layers: int, dropout_rate: float, use_bidirectional: bool) -> None:
                super().__init__()
                # 使用配置参数控制GRU结构
                self.gru = nn.GRU(
                    in_features, 
                    hidden, 
                    batch_first=True, 
                    num_layers=num_layers, 
                    dropout=dropout_rate,
                    bidirectional=use_bidirectional
                )
                # 计算双向GRU的输出维度
                gru_output_size = hidden * 2 if use_bidirectional else hidden
                
                # 增加一个中间层提高模型复杂度
                self.intermediate = nn.Linear(gru_output_size, hidden)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(dropout_rate)
                self.head = nn.Linear(hidden, out_features)
                
                # 添加层归一化
                self.layer_norm = nn.LayerNorm(gru_output_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.gru(x)
                last = out[:, -1, :]
                # 应用层归一化
                normalized = self.layer_norm(last)
                # 通过中间层
                intermediate = self.intermediate(normalized)
                activated = self.activation(intermediate)
                dropped = self.dropout(activated)
                return self.head(dropped)

        return GRUNet(
            input_size,
            self.config.hidden_units,
            output_size,
            num_layers,
            dropout_rate,
            use_bidirectional,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_dates: pd.Index | None = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Path]:
        """训练模型，仅负责输出模型与日志文件。"""
        X, y, fit_sample_dates = self._restrict_fit_window(X, y, sample_dates)
        model_type = self._model_type()
        panel_mode = X.ndim == 4 and y.ndim == 2 and X.shape[1] == y.shape[1]
        self.panel_mode = panel_mode
        loader_generator = self._configure_reproducibility()
        train_ratio = self._resolve_train_ratio()
        fit_date_start = pd.Timestamp(fit_sample_dates[0]).date() if fit_sample_dates is not None and len(fit_sample_dates) else None
        fit_date_end = pd.Timestamp(fit_sample_dates[-1]).date() if fit_sample_dates is not None and len(fit_sample_dates) else None

        if panel_mode:
            n_date_samples = X.shape[0]
            split = int(n_date_samples * train_ratio)
            if split <= 0 or split >= n_date_samples:
                split = max(n_date_samples - 1, 1)

            X_train_panel, X_val_panel = X[:split], X[split:]
            y_train_panel, y_val_panel = y[:split], y[split:]
            train_valid_mask = np.isfinite(y_train_panel)
            val_valid_mask = np.isfinite(y_val_panel)

            if not train_valid_mask.any():
                raise ValueError("No valid training panel samples remain after date-based split")
            if not val_valid_mask.any():
                raise ValueError("No valid validation panel samples remain after date-based split")

            X_train = X_train_panel[train_valid_mask]
            y_train = y_train_panel[train_valid_mask].astype(float).reshape(-1, 1)
            X_val = X_val_panel[val_valid_mask]
            y_val = y_val_panel[val_valid_mask].astype(float).reshape(-1, 1)
            train_date_start = pd.Timestamp(fit_sample_dates[0]).date() if fit_sample_dates is not None and len(fit_sample_dates) else None
            train_date_end = pd.Timestamp(fit_sample_dates[split - 1]).date() if fit_sample_dates is not None and split > 0 else None
            val_date_start = pd.Timestamp(fit_sample_dates[split]).date() if fit_sample_dates is not None and split < len(fit_sample_dates) else None
            val_date_end = pd.Timestamp(fit_sample_dates[-1]).date() if fit_sample_dates is not None and len(fit_sample_dates) else None

            n_samples = X_train.shape[0] + X_val.shape[0]
            seq_len = X.shape[2]
            self.seq_len = seq_len
            self.feature_dim = X.shape[3]
            X_model = X_train
            y_model = y_train
        else:
            n_samples = X.shape[0]
            seq_len = X.shape[1]
            self.seq_len = seq_len
            self.feature_dim = X.shape[2] * X.shape[3]
            X_model = X.reshape(n_samples, seq_len, -1)
            y_model = y
            train_date_start = fit_date_start
            train_date_end = None
            val_date_start = None
            val_date_end = fit_date_end

        if model_type == "lr":
            if Ridge is None:
                raise ImportError("model_type=lr requires scikit-learn to be installed")
            train_sample_count = int(X_model.shape[0])
            X2 = X_model.reshape(train_sample_count, -1)
            if panel_mode:
                val_sample_count = int(X_val.shape[0])
            else:
                split = int(n_samples * train_ratio)
                if split <= 0 or split >= n_samples:
                    split = max(n_samples - 1, 1)
                val_sample_count = max(int(n_samples - split), 0)
                train_date_end = pd.Timestamp(fit_sample_dates[split - 1]).date() if fit_sample_dates is not None and split > 0 else None
                val_date_start = pd.Timestamp(fit_sample_dates[split]).date() if fit_sample_dates is not None and split < len(fit_sample_dates) else None
            lr_model = Ridge(
                alpha=float(getattr(self.config, "lr_alpha", 1.0) or 1.0),
                fit_intercept=bool(getattr(self.config, "lr_fit_intercept", True)),
                max_iter=int(getattr(self.config, "lr_max_iter", 2000) or 2000),
            )
            lr_model.fit(X2, y_model.ravel() if panel_mode else y_model)
            self.model = lr_model
            outputs: Dict[str, Path] = {}
            if save_artifacts:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = Path(self.config.model_output_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"lr_model_{timestamp}.pkl"
                latest_path = model_dir / "lr_model_latest.pkl"
                with model_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": lr_model,
                            "seq_len": self.seq_len,
                            "feature_dim": self.feature_dim,
                            "panel_mode": self.panel_mode,
                        },
                        handle,
                    )
                with latest_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": lr_model,
                            "seq_len": self.seq_len,
                            "feature_dim": self.feature_dim,
                            "panel_mode": self.panel_mode,
                        },
                        handle,
                    )

                log_dir = Path(self.config.log_output_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"lr_training_log_{timestamp}.csv"
                pd.DataFrame(
                    [
                        {
                            "model_type": "lr",
                            "train_samples": train_sample_count,
                            "val_samples": val_sample_count,
                            "train_ratio": train_ratio,
                            "fit_window_months": int(getattr(self.config, "fit_window_months", 0) or 0),
                            "fit_date_start": fit_date_start,
                            "fit_date_end": fit_date_end,
                            "train_date_start": train_date_start,
                            "train_date_end": train_date_end,
                            "val_date_start": val_date_start,
                            "val_date_end": val_date_end,
                            "feature_dim": int(X2.shape[1]),
                            "target_dim": 1 if panel_mode else int(y_model.shape[1]),
                            "random_seed": int(getattr(self.config, "random_seed", 42) or 42),
                            "deterministic_training": bool(getattr(self.config, "deterministic_training", True)),
                        }
                    ]
                ).to_csv(log_path, index=False)
                outputs = {"model_path": model_path, "latest_path": latest_path, "log_path": log_path}
            return outputs

        # 检测并使用 GPU（支持自动选择最空闲显卡）
        import os
        device = torch.device("cpu")

        if torch.cuda.is_available():
            env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            configured_device_value = getattr(self.config, "cuda_device", None)
            configured_device = -1 if configured_device_value is None else int(configured_device_value)

            if env_cuda:
                # 设置了环境变量时，进程可见设备会被重映射为从 0 开始
                device = torch.device("cuda:0")
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                print(
                    f"CUDA_VISIBLE_DEVICES={env_cuda}, using cuda:0 "
                    f"(free={free_bytes / 1024**3:.2f}GiB/{total_bytes / 1024**3:.2f}GiB)"
                )
            elif configured_device >= 0:
                # 显式固定设备
                device = torch.device(f"cuda:{configured_device}")
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                print(
                    f"Using fixed GPU[{configured_device}]: {torch.cuda.get_device_name(configured_device)} "
                    f"(free={free_bytes / 1024**3:.2f}GiB/{total_bytes / 1024**3:.2f}GiB)"
                )
            else:
                # 自动选择当前空闲显存最多的一张卡
                best_idx = 0
                best_free = -1
                for gpu_idx in range(torch.cuda.device_count()):
                    try:
                        free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
                    except Exception:
                        free_bytes = -1
                    if free_bytes > best_free:
                        best_free = free_bytes
                        best_idx = gpu_idx

                device = torch.device(f"cuda:{best_idx}")
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                print(
                    f"Auto-selected GPU[{best_idx}]: {torch.cuda.get_device_name(best_idx)} "
                    f"(free={free_bytes / 1024**3:.2f}GiB/{total_bytes / 1024**3:.2f}GiB)"
                )
        else:
            print("Using CPU")

        if not panel_mode:
            split = int(n_samples * train_ratio)
            if split <= 0 or split >= n_samples:
                split = max(n_samples - 1, 1)
            X_train, X_val = X_model[:split], X_model[split:]
            y_train, y_val = y_model[:split], y_model[split:]
            train_date_end = pd.Timestamp(fit_sample_dates[split - 1]).date() if fit_sample_dates is not None and split > 0 else None
            val_date_start = pd.Timestamp(fit_sample_dates[split]).date() if fit_sample_dates is not None and split < len(fit_sample_dates) else None

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=loader_generator,
        )

        model_output_dim = 1 if panel_mode else y_model.shape[1]
        self.model = self.build_model(X_model.shape[2], model_output_dim,
                                      self.config.num_layers,
                                      self.config.dropout_rate,
                                      self.config.use_bidirectional)
        # 移动模型到GPU
        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        log_records: List[Dict[str, float]] = []
        
        # 添加早停机制
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict: Dict[str, torch.Tensor] | None = None
        patience_counter = 0
        patience_limit = 10
        
        # 优化：在循环外预先创建验证集 DataLoader，避免每轮迭代反复创建导致 OOM 和内存碎片
        val_loader = None
        if len(X_val) > 0:
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        epoch_iterator = tqdm(range(self.config.epochs), desc="GRU训练", unit="epoch")
        for epoch in epoch_iterator:
            self.model.train()
            batch_losses: List[float] = []
            for batch_X, batch_y in loader:
                # 移动数据到GPU
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

            val_loss = np.nan
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_batch_losses = []
                    for v_batch_X, v_batch_y in val_loader:
                        v_batch_X = v_batch_X.to(device)
                        v_batch_y = v_batch_y.to(device)
                        v_pred = self.model(v_batch_X)
                        v_loss = criterion(v_pred, v_batch_y)
                        val_batch_losses.append(v_loss.item())
                        # 显式清除显存
                        del v_batch_X, v_batch_y, v_pred, v_loss
                    val_loss = float(np.mean(val_batch_losses))
                
                # 更新学习率调度器
                scheduler.step(val_loss)

            current_lr = float(optimizer.param_groups[0]["lr"])
            if np.isnan(val_loss):
                epoch_iterator.set_postfix(train_loss=f"{train_loss:.6f}", lr=f"{current_lr:.2e}")
            else:
                epoch_iterator.set_postfix(
                    train_loss=f"{train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                    lr=f"{current_lr:.2e}",
                    patience=f"{patience_counter}/{patience_limit}",
                )

            log_records.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                assert self.model is not None
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                if save_artifacts:
                    model_dir = Path(self.config.model_output_dir)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    best_model_path = model_dir / "gru_model_best.pt"
                    torch.save(
                        {
                            "state_dict": best_state_dict,
                            "seq_len": self.seq_len,
                            "feature_dim": self.feature_dim,
                            "panel_mode": self.panel_mode,
                        },
                        best_model_path,
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        epoch_iterator.close()

        if best_state_dict is not None and self.model is not None:
            self.model.load_state_dict(best_state_dict)

        outputs: Dict[str, Path] = {}
        if save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = Path(self.config.model_output_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"gru_model_{timestamp}.pt"
            latest_path = model_dir / "gru_model_latest.pt"
            assert self.model is not None  # for type checkers
            payload = {
                "state_dict": self.model.state_dict(),
                "seq_len": self.seq_len,
                "feature_dim": self.feature_dim,
                "panel_mode": self.panel_mode,
            }
            torch.save(payload, model_path)
            torch.save(payload, latest_path)

            log_dir = Path(self.config.log_output_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"gru_training_log_{timestamp}.csv"
            log_df = pd.DataFrame(log_records)
            log_df["random_seed"] = int(getattr(self.config, "random_seed", 42) or 42)
            log_df["deterministic_training"] = bool(getattr(self.config, "deterministic_training", True))
            log_df["panel_mode"] = bool(panel_mode)
            log_df["validation_split_mode"] = "date_block" if panel_mode else "sample_tail"
            log_df["train_ratio"] = train_ratio
            log_df["fit_window_months"] = int(getattr(self.config, "fit_window_months", 0) or 0)
            log_df["fit_date_start"] = fit_date_start
            log_df["fit_date_end"] = fit_date_end
            log_df["train_date_start"] = train_date_start
            log_df["train_date_end"] = train_date_end
            log_df["val_date_start"] = val_date_start
            log_df["val_date_end"] = val_date_end
            log_df["train_samples"] = int(len(X_train))
            log_df["val_samples"] = int(len(X_val))
            log_df.to_csv(log_path, index=False)

            outputs = {"model_path": model_path, "latest_path": latest_path, "log_path": log_path}

        if best_epoch > 0:
            print(f"Best validation epoch: {best_epoch}, best_val_loss={best_val_loss:.6f}")

        # 训练结束后，将模型移回 CPU，并清空 GPU 缓存防止显存泄漏（特别是循环回测时）
        if self.model is not None:
            self.model = self.model.to("cpu")
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler' in locals():
            del scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs

    def predict_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """对批量序列执行预测，返回未来收益估计。"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        model_type = self._model_type()
        if self.panel_mode:
            if sequences.ndim == 3:
                sequences = sequences[np.newaxis, ...]
            if sequences.ndim != 4:
                raise ValueError("Expected input with dimensions (batch, stocks, seq_len, features) in panel mode")
            if self.seq_len is not None and sequences.shape[2] != self.seq_len:
                raise ValueError("Sequence length mismatch for panel prediction")
            if self.feature_dim is not None and sequences.shape[3] != self.feature_dim:
                raise ValueError("Feature dimension mismatch for panel prediction")

            batch_size, stock_count, _, _ = sequences.shape
            panel_batch = sequences.reshape(batch_size * stock_count, sequences.shape[2], sequences.shape[3])
            if model_type == "lr":
                X2 = panel_batch.reshape(panel_batch.shape[0], -1)
                preds = np.asarray(self.model.predict(X2), dtype=float).reshape(batch_size, stock_count)
                return preds

            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                tensor = torch.tensor(panel_batch, dtype=torch.float32).to(device)
                predictions = self.model(tensor).cpu().numpy().reshape(batch_size, stock_count)
                return predictions

        if sequences.ndim == 3:
            sequences = sequences[np.newaxis, ...]
        if sequences.ndim != 4:
            raise ValueError("Expected input with dimensions (batch, seq_len, stocks, features)")
        if self.seq_len is not None and sequences.shape[1] != self.seq_len:
            raise ValueError("Sequence length mismatch for prediction")
        if self.feature_dim is not None and sequences.shape[2] * sequences.shape[3] != self.feature_dim:
            raise ValueError("Feature dimension mismatch for prediction")
        batch = sequences.reshape(sequences.shape[0], sequences.shape[1], -1)

        if model_type == "lr":
            X2 = batch.reshape(batch.shape[0], -1)
            return np.asarray(self.model.predict(X2), dtype=float)

        self.model.eval()

        # 确定设备
        device = next(self.model.parameters()).device

        with torch.no_grad():
            tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            predictions = self.model(tensor)
            return predictions.cpu().numpy()

    def predict_with_uncertainty(self, sequences: np.ndarray, n_samples: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """使用蒙特卡洛dropout进行不确定性估计"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        if self._model_type() == "lr":
            mean_pred = self.predict_sequences(sequences)
            std_pred = np.zeros_like(mean_pred, dtype=float)
            return mean_pred, std_pred
        if self.panel_mode:
            if sequences.ndim == 3:
                sequences = sequences[np.newaxis, ...]
            panel_batch = sequences.reshape(sequences.shape[0] * sequences.shape[1], sequences.shape[2], sequences.shape[3])
            sample_count = max(1, int(n_samples or self.config.uncertainty_samples or 10))
            self.model.eval()
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            device = next(self.model.parameters()).device
            tensor = torch.tensor(panel_batch, dtype=torch.float32).to(device)
            predictions = []
            with torch.no_grad():
                for _ in range(sample_count):
                    pred = self.model(tensor)
                    predictions.append(pred.cpu().numpy().reshape(sequences.shape[0], sequences.shape[1]))
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            return mean_pred, std_pred
        if sequences.ndim == 3:
            sequences = sequences[np.newaxis, ...]
        batch = sequences.reshape(sequences.shape[0], sequences.shape[1], -1)
        sample_count = max(1, int(n_samples or self.config.uncertainty_samples or 10))
        
        self.model.eval()
        # 启用dropout进行不确定性评估
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        device = next(self.model.parameters()).device
        tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        
        predictions = []
        with torch.no_grad():
            for _ in range(sample_count):
                pred = self.model(tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 恢复dropout为eval模式
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
                
        return mean_pred, std_pred

    def predict(self, latest_sequence: np.ndarray) -> np.ndarray:
        """便捷封装：对单个序列执行预测。"""
        return self.predict_sequences(latest_sequence)[0]

    def prepare_latest_sequence(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        expressions: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """提取最新一段序列用于推理。"""
        factors = self._build_factor_tensor(price_df, factor_dict, expressions)
        if factors.shape[0] < self.config.sequence_length:
            raise ValueError("样本长度不足以构造最新预测序列，请检查配置。")
        feature_valid_mask = self._build_feature_availability_mask(price_df, factor_dict)
        latest_valid = feature_valid_mask.iloc[-self.config.sequence_length :].all(axis=0)
        self._latest_sample_valid_mask = pd.DataFrame(
            [latest_valid.to_numpy(dtype=bool)],
            index=pd.DatetimeIndex([pd.Timestamp(price_df.index[-1])]),
            columns=price_df.columns,
        )
        if self.panel_mode:
            latest_sequence = np.transpose(factors[-self.config.sequence_length :], (1, 0, 2))
        else:
            latest_sequence = factors[-self.config.sequence_length :]
        return latest_sequence, list(price_df.columns)

    def load_model(
        self,
        latest_sequence: np.ndarray,
        stock_names: List[str],
        model_path: str | Path | None = None,
    ) -> None:
        """加载已训练好的模型权重，用于独立预测流程。"""
        model_type = self._model_type()
        default_name = "lr_model_latest.pkl" if model_type == "lr" else "gru_model_latest.pt"
        checkpoint = Path(model_path or Path(self.config.model_output_dir) / default_name)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if latest_sequence.ndim == 3:
            self.seq_len = latest_sequence.shape[1]
            self.feature_dim = latest_sequence.shape[2]
        else:
            self.seq_len = latest_sequence.shape[1]
            self.feature_dim = latest_sequence.shape[2] * latest_sequence.shape[3]
        if model_type == "lr":
            with checkpoint.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.seq_len = int(payload.get("seq_len", self.seq_len))
                self.feature_dim = int(payload.get("feature_dim", self.feature_dim))
                self.panel_mode = bool(payload.get("panel_mode", self.panel_mode))
            else:
                self.model = payload
            if self.model is None:
                raise RuntimeError("Failed to load LR model payload")
            return

        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(payload, dict):
            self.seq_len = int(payload.get("seq_len", self.seq_len))
            self.feature_dim = int(payload.get("feature_dim", self.feature_dim))
            self.panel_mode = bool(payload.get("panel_mode", self.panel_mode))
        output_dim = 1 if self.panel_mode else len(stock_names)
        self.model = self.build_model(self.feature_dim, output_dim)
        
        # 处理模型结构不匹配的问题（例如注意力机制相关参数）
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        # 加载匹配的参数
        self.model.load_state_dict(filtered_state_dict, strict=False)

    def recommend(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        expressions: List[str],
        model_path: str | Path | None = None,
        top_n: int | None = None,
        output_dir: str | None = None,
        save_file: bool = True,
    ) -> Tuple[pd.Series, Path | None]:
        """使用已有模型对最新数据排序，并可选保存推荐结果。"""
        latest_sequence, stock_names = self.prepare_latest_sequence(price_df, factor_dict, expressions)
        if self.model is None:
            self.load_model(latest_sequence, stock_names, model_path)
        predictions = self.predict(latest_sequence)
        series = pd.Series(predictions, index=stock_names)
        if self._latest_sample_valid_mask is not None:
            valid_series = self._latest_sample_valid_mask.iloc[-1].reindex(stock_names).fillna(False)
            series = series.where(valid_series)
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        # 直接按预测得分排序选股
        ranked = series.sort_values(ascending=False)
        
        chosen_top = top_n or self.config.top_n or min(10, len(ranked))
        recommendation_path: Path | None = None
        if save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 尝试从数据文件名中提取日期
            data_date = ""
            if self.config.price_file:
                import re
                match = re.search(r"(\d{4}-\d{2}-\d{2})", str(self.config.price_file))
                if match:
                    data_date = match.group(1).replace("-", "") + "_"
            
            out_dir = Path(output_dir or self.config.recommendation_output_dir or self.config.log_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            recommendation_path = out_dir / f"gru_recommendations_{data_date}{timestamp}.csv"
            # 保存原始预测值
            df_out = pd.DataFrame({"score": series})
            # 根据预测得分降序排序
            df_out.sort_values("score", ascending=False).to_csv(recommendation_path)

        # 返回按原始预测排序的 Series（保留原始值），以及推荐文件路径
        return ranked, recommendation_path

    def recommend_with_confidence(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        expressions: List[str],
        model_path: str | Path | None = None,
        top_n: int | None = None,
        output_dir: str | None = None,
        save_file: bool = True,
    ) -> Tuple[pd.DataFrame, Path | None]:
        """使用不确定性估计的推荐函数，返回包含置信度的推荐结果。"""
        latest_sequence, stock_names = self.prepare_latest_sequence(price_df, factor_dict, expressions)
        if self.model is None:
            self.load_model(latest_sequence, stock_names, model_path)
        
        # 获取预测均值和标准差
        mean_pred, std_pred = self.predict_with_uncertainty(latest_sequence)
        
        # 计算置信度（使用负的标准差，因为标准差越小置信度越高）
        confidence = 1 / (1 + std_pred)
        
        # 创建包含预测值和置信度的DataFrame
        result_df = pd.DataFrame({
            'prediction': mean_pred,
            'uncertainty': std_pred,
            'confidence': confidence
        }, index=stock_names)
        if self._latest_sample_valid_mask is not None:
            valid_series = self._latest_sample_valid_mask.iloc[-1].reindex(stock_names).fillna(False)
            result_df = result_df.loc[valid_series]
        
        # 按预测值排序
        result_df = result_df.sort_values('prediction', ascending=False)
        
        chosen_top = top_n or self.config.top_n or min(10, len(result_df))
        recommendation_path: Path | None = None
        if save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(output_dir or self.config.recommendation_output_dir or self.config.log_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            recommendation_path = out_dir / f"gru_recommendations_with_confidence_{timestamp}.csv"
            result_df.head(chosen_top).to_csv(recommendation_path)
            
        return result_df, recommendation_path

    def _build_factor_tensor(
        self,
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        expressions: List[str],
    ) -> np.ndarray:
        """计算并堆叠 GA 表达式对应的多维因子张量。"""
        data_sources = self._build_expression_data_sources(price_df, factor_dict)

        computed_factors: List[pd.DataFrame] = []
        skipped_expressions: List[str] = []
        for expression in expressions:
            try:
                computed = self._evaluate_expression(expression, data_sources)
                computed = computed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                computed_factors.append(computed)
            except KeyError as exc:
                skipped_expressions.append(f"{expression} (missing: {exc})")
                continue

        if not computed_factors:
            details = "\n".join(skipped_expressions[:10])
            raise ValueError(
                "在因子库中未找到任何可用的表达式，请检查因子名称是否正确。"
                + (f"\n跳过的表达式示例:\n{details}" if details else "")
            )

        if skipped_expressions:
            print(
                f"Warning: skipped {len(skipped_expressions)} expressions due to missing factors. "
                f"Using {len(computed_factors)} valid expressions."
            )

        factors = np.stack([frame.values for frame in computed_factors], axis=-1)
        factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)

        # 可选：按日期截面对每个因子做 z-score 标准化，消除不同股票绝对量纲的影响
        if getattr(self.config, 'normalize_factors', False):
            # factors shape: (dates, stocks, factors)
            # 计算每个日期（axis=1就是 stocks 轴）上的均值和 std
            means = factors.mean(axis=1, keepdims=True)  # shape (dates,1,factors)
            stds = factors.std(axis=1, keepdims=True)
            # 避免除以0
            stds[stds == 0] = 1.0
            factors = (factors - means) / stds

        return factors


def load_gru_config() -> GRUConfig:
    cfg = get_config_value("gru_pipeline")
    ga_cfg = get_config_value("ga_pipeline")
    return GRUConfig(
        sequence_length=cfg.get("sequence_length", 20),
        predict_days=cfg.get("predict_days", 10),
        hidden_units=cfg.get("hidden_units", 64),
        batch_size=cfg.get("batch_size", 32),
        epochs=cfg.get("epochs", 50),
        validation_ratio=cfg.get("validation_ratio", 0.2),
        train_ratio=cfg.get("train_ratio"),
        factor_expression_file=cfg.get("factor_expression_file"),
        price_file=cfg.get("price_file"),
        factor_file=cfg.get("factor_file"),
        model_output_dir=cfg.get("model_output_dir", "/home/intership/guojh/GA和GRU/models"),
        log_output_dir=cfg.get("log_output_dir", "/home/intership/guojh/GA和GRU/reports"),
        backtest_price_file=cfg.get("backtest_price_file"),
        backtest_factor_file=cfg.get("backtest_factor_file"),
        recommendation_output_dir=cfg.get("recommendation_output_dir"),
        top_n=cfg.get("top_n"),
        num_layers=cfg.get("num_layers", 3),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_bidirectional=cfg.get("use_bidirectional", True),
        cuda_device=cfg.get("cuda_device", 0),
        normalize_factors=cfg.get("normalize_factors", True),
        normalize_target=cfg.get("normalize_target", True),
        uncertainty_samples=cfg.get("uncertainty_samples", 10),
        random_seed=int(cfg.get("random_seed", 42)),
        deterministic_training=bool(cfg.get("deterministic_training", True)),
        target_mode=str(cfg.get("target_mode", "legacy_price_return")),
        target_parquet_root=cfg.get("target_parquet_root"),
        target_backtest_parquet_root=cfg.get("target_backtest_parquet_root"),
        target_train_field=str(cfg.get("target_train_field", "neu_all_vwap_5d")),
        target_backtest_field=str(cfg.get("target_backtest_field", "raw_vwap_5d")),
        signal_to_entry_days=int(cfg.get("signal_to_entry_days", 1)),
        target_horizon_days=int(cfg.get("target_horizon_days", 5)),
        backtest_return_mode=str(cfg.get("backtest_return_mode", "daily_nextday")),
        enable_3d_mining=bool(cfg.get("enable_3d_mining", ga_cfg.get("enable_3d_mining", False))),
        minute_cache_root=cfg.get("minute_cache_root", ga_cfg.get("minute_cache_root")),
        minute_field_names=tuple(cfg.get("minute_field_names", ga_cfg.get("minute_field_names", ()))) or None,
        minute_field_prefix=str(cfg.get("minute_field_prefix", ga_cfg.get("minute_field_prefix", "min_"))),
        ga_3d_dtype=str(cfg.get("ga_3d_dtype", ga_cfg.get("ga_3d_dtype", "float32"))),
        model_type=str(cfg.get("model_type", "gru")),
        lr_alpha=float(cfg.get("lr_alpha", 1.0)),
        lr_fit_intercept=bool(cfg.get("lr_fit_intercept", True)),
        lr_max_iter=int(cfg.get("lr_max_iter", 2000)),
        fit_window_months=cfg.get("fit_window_months"),
    )
