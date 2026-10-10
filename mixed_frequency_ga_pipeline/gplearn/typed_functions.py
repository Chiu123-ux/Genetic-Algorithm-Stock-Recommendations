"""Typed function registry for the 3D-native minute GP prototype.

This module is intentionally separate from gplearn.functions so the first
implementation step can introduce reducer-aware metadata without destabilising
the existing daily GP workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


MINUTE_TENSOR = "minute_tensor"
DAILY_MATRIX = "daily_matrix"


@dataclass(frozen=True)
class TypedFunction:
    name: str
    arity: int
    input_types: Tuple[str, ...]
    output_type: str
    function: Callable[..., np.ndarray]

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.function(*args)


def _same_shape(x1: np.ndarray, x2: np.ndarray) -> bool:
    return tuple(x1.shape) == tuple(x2.shape)


def _is_daily_runtime_value(x1: np.ndarray) -> bool:
    return x1.ndim in (2, 3)


def _protected_division(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x2) > 1e-6, np.divide(x1, x2), 0.0)


def _protected_log(x1: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 1e-6, np.log(np.abs(x1)), 0.0)


def _protected_sqrt(x1: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x1))


def _protected_inverse(x1: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 1e-6, 1.0 / x1, 0.0)


def _protected_sigmoid(x1: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-x1))


def _protected_tanh(x1: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.tanh(x1)


def _signedpower(x1: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(clean) * np.square(np.abs(clean))


def _abssqrt(x1: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x1))


def _sign(x1: np.ndarray) -> np.ndarray:
    out = np.sign(x1)
    return np.where(np.isnan(x1), np.nan, out)


def _diff(x1: np.ndarray) -> np.ndarray:
    if x1.shape[-1] == 0:
        return np.empty_like(x1)
    out = np.diff(x1, axis=-1, prepend=np.nan)
    return out


def _zscore(x1: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x1, axis=-1, keepdims=True)
    std = np.nanstd(x1, axis=-1, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x1 - mean) / std


def _delay(x1: np.ndarray, steps: int) -> np.ndarray:
    if x1.shape[-1] == 0:
        return np.empty_like(x1)

    lag = max(1, int(steps))
    out = np.full(x1.shape, np.nan, dtype=float)
    if lag >= x1.shape[-1]:
        return out
    out[..., lag:] = x1[..., :-lag]
    return out


def _day_delay(x1: np.ndarray, steps: int) -> np.ndarray:
    if not _is_daily_runtime_value(x1):
        raise ValueError(f"day_delay expects daily runtime value with ndim in (2, 3), got shape={x1.shape}")
    if x1.shape[0] == 0:
        return np.empty_like(x1)

    lag = max(1, int(steps))
    out = np.full(x1.shape, np.nan, dtype=float)
    if lag >= x1.shape[0]:
        return out
    out[lag:, ...] = x1[:-lag, ...]
    return out


def _delta(x1: np.ndarray, steps: int) -> np.ndarray:
    delayed = _delay(x1, steps=steps)
    return x1 - delayed


def _day_delta(x1: np.ndarray, steps: int) -> np.ndarray:
    delayed = _day_delay(x1, steps=steps)
    return x1 - delayed


def _rolling_preserve(window: int, reducer: Callable[..., np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        if x1.shape[-1] == 0:
            return np.empty_like(x1)

        width = min(max(int(window), 1), x1.shape[-1])
        out = np.full(x1.shape, np.nan, dtype=float)
        rolled = np.lib.stride_tricks.sliding_window_view(x1, window_shape=width, axis=-1)
        reduced = reducer(rolled, axis=-1)
        out[..., width - 1 :] = reduced
        return out

    return _inner


def _rolling_daily(window: int, reducer: Callable[..., np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        if not _is_daily_runtime_value(x1):
            raise ValueError(f"daily rolling expects daily runtime value with ndim in (2, 3), got shape={x1.shape}")
        if x1.shape[0] == 0:
            return np.empty_like(x1)

        width = min(max(int(window), 1), x1.shape[0])
        out = np.full(x1.shape, np.nan, dtype=float)
        rolled = np.lib.stride_tricks.sliding_window_view(x1, window_shape=width, axis=0)
        reduced = reducer(rolled, axis=-1)
        out[width - 1 :, ...] = reduced
        return out

    return _inner


def _wrap_binary_daily(op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if not _same_shape(x1, x2):
            raise ValueError(f"daily binary op requires same-shape inputs: {x1.shape} vs {x2.shape}")
        if not _is_daily_runtime_value(x1) or not _is_daily_runtime_value(x2):
            raise ValueError(f"daily binary op expects daily runtime values with ndim in (2, 3), got {x1.shape} vs {x2.shape}")
        return op(x1, x2)

    return _inner


def _wrap_unary_daily(op: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        if not _is_daily_runtime_value(x1):
            raise ValueError(f"daily unary op expects daily runtime value with ndim in (2, 3), got shape={x1.shape}")
        return op(x1)

    return _inner


def _wrap_binary_preserve(function: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if not _same_shape(x1, x2):
            raise ValueError("typed preserve function requires same-shape inputs")
        return function(x1, x2)

    return _inner


def _reduce_last_valid_count(mask: np.ndarray) -> np.ndarray:
    return np.sum(mask, axis=-1)


def _nanmean_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nanmean(x1, axis=-1)


def _nanstd_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nanstd(x1, axis=-1)


def _nanmax_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nanmax(x1, axis=-1)


def _nanmin_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nanmin(x1, axis=-1)


def _sum_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nansum(x1, axis=-1)


def _last_reduce(x1: np.ndarray) -> np.ndarray:
    if x1.shape[-1] == 0:
        return np.zeros(x1.shape[:-1], dtype=float)
    return x1[..., -1]


def _range_reduce(x1: np.ndarray) -> np.ndarray:
    return np.nanmax(x1, axis=-1) - np.nanmin(x1, axis=-1)


def _argmax_reduce(x1: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x1)
    safe = np.where(valid, x1, -np.inf)
    out = np.argmax(safe, axis=-1).astype(float)
    return np.where(valid.any(axis=-1), out, 0.0)


def _argmin_reduce(x1: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x1)
    safe = np.where(valid, x1, np.inf)
    out = np.argmin(safe, axis=-1).astype(float)
    return np.where(valid.any(axis=-1), out, 0.0)


def _percentile_reduce(percentile: float) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        return np.nanpercentile(x1, percentile, axis=-1)

    return _inner


def _window_slice(x1: np.ndarray, mode: str, window: int) -> np.ndarray:
    width = min(max(int(window), 1), x1.shape[-1]) if x1.shape[-1] > 0 else 0
    if width == 0:
        return x1[..., :0]
    if mode == "head":
        return x1[..., :width]
    if mode == "tail":
        return x1[..., -width:]
    raise ValueError(f"Unsupported window mode: {mode}")


def _window_reducer(mode: str, window: int, reducer: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        return reducer(_window_slice(x1, mode=mode, window=window))

    return _inner


def _segment_slice(x1: np.ndarray, start_ratio: float, end_ratio: float) -> np.ndarray:
    length = x1.shape[-1]
    if length <= 0:
        return x1[..., :0]
    start = int(np.floor(length * float(start_ratio)))
    end = int(np.floor(length * float(end_ratio)))
    start = max(0, min(start, length - 1))
    end = max(start + 1, min(end, length))
    return x1[..., start:end]


def _half_slice(x1: np.ndarray, session: str) -> np.ndarray:
    length = x1.shape[-1]
    if length <= 0:
        return x1[..., :0]
    split = (length + 1) // 2
    if session == "am":
        return x1[..., :split]
    if session == "pm":
        return x1[..., split:]
    raise ValueError(f"Unsupported session: {session}")


def _half_reducer(session: str, reducer: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        return reducer(_half_slice(x1, session=session))

    return _inner


def _segment_reducer(start_ratio: float, end_ratio: float, reducer: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _inner(x1: np.ndarray) -> np.ndarray:
        return reducer(_segment_slice(x1, start_ratio=start_ratio, end_ratio=end_ratio))

    return _inner


def _segment_pair_reduce(
    x1: np.ndarray,
    x2: np.ndarray,
    start_ratio: float,
    end_ratio: float,
    reducer: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    return reducer(
        _segment_slice(x1, start_ratio=start_ratio, end_ratio=end_ratio),
        _segment_slice(x2, start_ratio=start_ratio, end_ratio=end_ratio),
    )


def _half_pair_reduce(
    x1: np.ndarray,
    x2: np.ndarray,
    session: str,
    reducer: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    return reducer(_half_slice(x1, session=session), _half_slice(x2, session=session))


def _am_pm_spread_reduce(x1: np.ndarray) -> np.ndarray:
    am_mean = _nanmean_reduce(_half_slice(x1, session="am"))
    pm_mean = _nanmean_reduce(_half_slice(x1, session="pm"))
    return pm_mean - am_mean


def _midday_reversal_reduce(x1: np.ndarray) -> np.ndarray:
    morning_tail = _nanmean_reduce(_segment_slice(x1, start_ratio=0.25, end_ratio=0.5))
    afternoon_head = _nanmean_reduce(_segment_slice(x1, start_ratio=0.5, end_ratio=0.75))
    return afternoon_head - morning_tail


def _session_beta_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return _half_pair_reduce(x1, x2, session="am", reducer=_beta_reduce)


def _afternoon_beta_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return _half_pair_reduce(x1, x2, session="pm", reducer=_beta_reduce)


def _pairwise_mask(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.isfinite(x1) & np.isfinite(x2)


def _cov_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if not _same_shape(x1, x2):
        raise ValueError("cov reducer requires same-shape inputs")
    mask = _pairwise_mask(x1, x2)
    count = np.maximum(_reduce_last_valid_count(mask), 1)
    safe_x1 = np.where(mask, x1, 0.0)
    safe_x2 = np.where(mask, x2, 0.0)
    mean_x1 = np.sum(safe_x1, axis=-1) / count
    mean_x2 = np.sum(safe_x2, axis=-1) / count
    centered_x1 = np.where(mask, x1 - mean_x1[..., None], 0.0)
    centered_x2 = np.where(mask, x2 - mean_x2[..., None], 0.0)
    return np.sum(centered_x1 * centered_x2, axis=-1) / count


def _corr_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    cov = _cov_reduce(x1, x2)
    std_x1 = np.nanstd(np.where(_pairwise_mask(x1, x2), x1, np.nan), axis=-1)
    std_x2 = np.nanstd(np.where(_pairwise_mask(x1, x2), x2, np.nan), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((std_x1 > 1e-12) & (std_x2 > 1e-12), cov / (std_x1 * std_x2), 0.0)


def _beta_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    cov = _cov_reduce(x1, x2)
    var_x2 = np.nanvar(np.where(_pairwise_mask(x1, x2), x2, np.nan), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var_x2 > 1e-12, cov / var_x2, 0.0)


def _weighted_mean_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if not _same_shape(x1, x2):
        raise ValueError("weighted_mean reducer requires same-shape inputs")
    mask = _pairwise_mask(x1, x2)
    weights = np.where(mask, x2, 0.0)
    numerator = np.sum(np.where(mask, x1 * x2, 0.0), axis=-1)
    denominator = np.sum(weights, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denominator) > 1e-12, numerator / denominator, 0.0)


def _mean_spread_reduce(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if not _same_shape(x1, x2):
        raise ValueError("mean_spread reducer requires same-shape inputs")
    return np.nanmean(x1 - x2, axis=-1)


def _daily_absdiff(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.abs(x1 - x2)


def _daily_norm_spread(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    denom = np.abs(x1) + np.abs(x2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-6, (x1 - x2) / denom, 0.0)


def _register(
    name: str,
    arity: int,
    input_types: Tuple[str, ...],
    output_type: str,
    function: Callable[..., np.ndarray],
) -> TypedFunction:
    return TypedFunction(
        name=name,
        arity=arity,
        input_types=input_types,
        output_type=output_type,
        function=function,
    )


TYPED_FUNCTION_MAP: Dict[str, TypedFunction] = {
    "add": _register("add", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(np.add)),
    "sub": _register("sub", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(np.subtract)),
    "mul": _register("mul", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(np.multiply)),
    "div": _register("div", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(_protected_division)),
    "abs": _register("abs", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, np.abs),
    "neg": _register("neg", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, np.negative),
    "log": _register("log", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _protected_log),
    "sqrt": _register("sqrt", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _protected_sqrt),
    "abssqrt": _register("abssqrt", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _abssqrt),
    "inv": _register("inv", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _protected_inverse),
    "sigmoid": _register("sigmoid", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _protected_sigmoid),
    "tanh": _register("tanh", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _protected_tanh),
    "max": _register("max", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(np.maximum)),
    "min": _register("min", 2, (MINUTE_TENSOR, MINUTE_TENSOR), MINUTE_TENSOR, _wrap_binary_preserve(np.minimum)),
    "sign": _register("sign", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _sign),
    "diff": _register("diff", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _diff),
    "zscore": _register("zscore", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _zscore),
    "signedpower": _register("signedpower", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _signedpower),
    "ts_delay_5": _register("ts_delay_5", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delay(x1, steps=5)),
    "ts_delay_10": _register("ts_delay_10", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delay(x1, steps=10)),
    "ts_delay_20": _register("ts_delay_20", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delay(x1, steps=20)),
    "ts_delta_5": _register("ts_delta_5", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delta(x1, steps=5)),
    "ts_delta_10": _register("ts_delta_10", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delta(x1, steps=10)),
    "ts_delta_20": _register("ts_delta_20", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, lambda x1: _delta(x1, steps=20)),
    "ts_mean_5": _register("ts_mean_5", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(5, np.nanmean)),
    "ts_mean_10": _register("ts_mean_10", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(10, np.nanmean)),
    "ts_mean_20": _register("ts_mean_20", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(20, np.nanmean)),
    "ts_std_5": _register("ts_std_5", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(5, np.nanstd)),
    "ts_std_10": _register("ts_std_10", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(10, np.nanstd)),
    "ts_std_20": _register("ts_std_20", 1, (MINUTE_TENSOR,), MINUTE_TENSOR, _rolling_preserve(20, np.nanstd)),
    "day_delay_5": _register("day_delay_5", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delay(x1, steps=5)),
    "day_delay_10": _register("day_delay_10", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delay(x1, steps=10)),
    "day_delay_20": _register("day_delay_20", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delay(x1, steps=20)),
    "day_delta_5": _register("day_delta_5", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delta(x1, steps=5)),
    "day_delta_10": _register("day_delta_10", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delta(x1, steps=10)),
    "day_delta_20": _register("day_delta_20", 1, (DAILY_MATRIX,), DAILY_MATRIX, lambda x1: _day_delta(x1, steps=20)),
    "day_mean_5": _register("day_mean_5", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(5, np.nanmean)),
    "day_mean_10": _register("day_mean_10", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(10, np.nanmean)),
    "day_mean_20": _register("day_mean_20", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(20, np.nanmean)),
    "day_std_5": _register("day_std_5", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(5, np.nanstd)),
    "day_std_10": _register("day_std_10", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(10, np.nanstd)),
    "day_std_20": _register("day_std_20", 1, (DAILY_MATRIX,), DAILY_MATRIX, _rolling_daily(20, np.nanstd)),
    "d_abs": _register("d_abs", 1, (DAILY_MATRIX,), DAILY_MATRIX, _wrap_unary_daily(np.abs)),
    "d_log": _register("d_log", 1, (DAILY_MATRIX,), DAILY_MATRIX, _wrap_unary_daily(_protected_log)),
    "d_sqrt": _register("d_sqrt", 1, (DAILY_MATRIX,), DAILY_MATRIX, _wrap_unary_daily(_protected_sqrt)),
    "d_sign": _register("d_sign", 1, (DAILY_MATRIX,), DAILY_MATRIX, _wrap_unary_daily(_sign)),
    "d_add": _register("d_add", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(np.add)),
    "d_sub": _register("d_sub", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(np.subtract)),
    "d_mul": _register("d_mul", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(np.multiply)),
    "d_div": _register("d_div", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(_protected_division)),
    "d_max": _register("d_max", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(np.maximum)),
    "d_min": _register("d_min", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(np.minimum)),
    "d_absdiff": _register("d_absdiff", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(_daily_absdiff)),
    "d_ratio": _register("d_ratio", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(_protected_division)),
    "d_norm_spread": _register("d_norm_spread", 2, (DAILY_MATRIX, DAILY_MATRIX), DAILY_MATRIX, _wrap_binary_daily(_daily_norm_spread)),
    "nanmean": _register("nanmean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _nanmean_reduce),
    "nanstd": _register("nanstd", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _nanstd_reduce),
    "nanmax": _register("nanmax", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _nanmax_reduce),
    "nanmin": _register("nanmin", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _nanmin_reduce),
    "sum": _register("sum", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _sum_reduce),
    "last": _register("last", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _last_reduce),
    "range": _register("range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _range_reduce),
    "argmax": _register("argmax", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _argmax_reduce),
    "argmin": _register("argmin", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _argmin_reduce),
    "q10": _register("q10", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _percentile_reduce(10)),
    "q50": _register("q50", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _percentile_reduce(50)),
    "q90": _register("q90", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _percentile_reduce(90)),
    "am_mean": _register("am_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("am", _nanmean_reduce)),
    "am_std": _register("am_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("am", _nanstd_reduce)),
    "am_range": _register("am_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("am", _range_reduce)),
    "pm_mean": _register("pm_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("pm", _nanmean_reduce)),
    "pm_std": _register("pm_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("pm", _nanstd_reduce)),
    "pm_range": _register("pm_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _half_reducer("pm", _range_reduce)),
    "mid_mean": _register("mid_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _segment_reducer(1.0 / 3.0, 2.0 / 3.0, _nanmean_reduce)),
    "mid_std": _register("mid_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _segment_reducer(1.0 / 3.0, 2.0 / 3.0, _nanstd_reduce)),
    "mid_range": _register("mid_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _segment_reducer(1.0 / 3.0, 2.0 / 3.0, _range_reduce)),
    "open15_mean": _register("open15_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 15, _nanmean_reduce)),
    "open15_std": _register("open15_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 15, _nanstd_reduce)),
    "open15_range": _register("open15_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 15, _range_reduce)),
    "open30_mean": _register("open30_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 30, _nanmean_reduce)),
    "open30_std": _register("open30_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 30, _nanstd_reduce)),
    "open30_sum": _register("open30_sum", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 30, _sum_reduce)),
    "open30_range": _register("open30_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("head", 30, _range_reduce)),
    "close15_mean": _register("close15_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 15, _nanmean_reduce)),
    "close15_std": _register("close15_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 15, _nanstd_reduce)),
    "close15_range": _register("close15_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 15, _range_reduce)),
    "close30_mean": _register("close30_mean", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 30, _nanmean_reduce)),
    "close30_std": _register("close30_std", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 30, _nanstd_reduce)),
    "close30_sum": _register("close30_sum", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 30, _sum_reduce)),
    "close30_range": _register("close30_range", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _window_reducer("tail", 30, _range_reduce)),
    "am_pm_spread": _register("am_pm_spread", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _am_pm_spread_reduce),
    "midday_reversal": _register("midday_reversal", 1, (MINUTE_TENSOR,), DAILY_MATRIX, _midday_reversal_reduce),
    "cov": _register("cov", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _cov_reduce),
    "corr": _register("corr", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _corr_reduce),
    "beta": _register("beta", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _beta_reduce),
    "am_beta": _register("am_beta", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _session_beta_reduce),
    "pm_beta": _register("pm_beta", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _afternoon_beta_reduce),
    "weighted_mean": _register("weighted_mean", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _weighted_mean_reduce),
    "mean_spread": _register("mean_spread", 2, (MINUTE_TENSOR, MINUTE_TENSOR), DAILY_MATRIX, _mean_spread_reduce),
}


MINUTE_PRESERVING_FUNCTIONS: Dict[str, TypedFunction] = {
    name: spec for name, spec in TYPED_FUNCTION_MAP.items() if spec.output_type == MINUTE_TENSOR
}


MINUTE_REDUCER_FUNCTIONS: Dict[str, TypedFunction] = {
    name: spec for name, spec in TYPED_FUNCTION_MAP.items() if spec.output_type == DAILY_MATRIX
}


DAILY_PRESERVING_FUNCTIONS: Dict[str, TypedFunction] = {
    name: spec
    for name, spec in TYPED_FUNCTION_MAP.items()
    if spec.output_type == DAILY_MATRIX and all(input_type == DAILY_MATRIX for input_type in spec.input_types)
}


def is_typed_function(node: object) -> bool:
    return isinstance(node, TypedFunction)


def is_minute_reducer(node: object) -> bool:
    return (
        isinstance(node, TypedFunction)
        and node.output_type == DAILY_MATRIX
        and all(input_type == MINUTE_TENSOR for input_type in node.input_types)
    )


def is_minute_preserving(node: object) -> bool:
    return isinstance(node, TypedFunction) and node.output_type == MINUTE_TENSOR


def is_daily_preserving(node: object) -> bool:
    return (
        isinstance(node, TypedFunction)
        and node.output_type == DAILY_MATRIX
        and all(input_type == DAILY_MATRIX for input_type in node.input_types)
    )