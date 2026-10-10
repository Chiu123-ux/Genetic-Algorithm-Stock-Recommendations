"""Utilities for building and loading per-day minute tensor caches.

The cache layout now prefers a mmap-friendly directory format:

- one NPY data file per trade date
- one NPY file per axis (stocks/minutes/fields/requested_fields)
- one JSON manifest per trade date

Legacy NPZ cache entries are still readable for backward compatibility.

Each cache entry stores a 3D array with shape:
    (n_stocks, n_minutes, n_fields)

The training pipeline can later stack daily entries into a logical 4D tensor:
    (n_dates, n_stocks, n_minutes, n_fields)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


DEFAULT_SOURCE_ROOT = "/raid0/intership/etfdata/Minutedata"
DEFAULT_CACHE_FORMAT = "npy"
DEFAULT_CACHE_FILENAME = "minute_tensor.npy"
LEGACY_CACHE_FILENAME = "minute_tensor.npz"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_STOCKS_FILENAME = "stocks.npy"
DEFAULT_MINUTES_FILENAME = "minutes.npy"
DEFAULT_FIELDS_FILENAME = "fields.npy"
DEFAULT_REQUESTED_FIELDS_FILENAME = "requested_fields.npy"
STOCK_COLUMN_CANDIDATES = ("Uid", "uid", "stock", "ticker", "symbol", "Code", "code")


@dataclass
class MinuteTensorCacheEntry:
    trade_date: str
    source_dir: str
    cache_file: str
    manifest_file: str
    dtype: str
    requested_fields: List[str]
    available_fields: List[str]
    missing_fields: List[str]
    stock_count: int
    minute_count: int
    cache_format: str = DEFAULT_CACHE_FORMAT
    data_file: str = ""
    stocks_file: str = ""
    minutes_file: str = ""
    fields_file: str = ""
    requested_fields_file: str = ""
    stock_filter_count: int = 0


def _normalise_trade_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalise_time_token(value: str | None) -> str | None:
    if value is None:
        return None

    token = str(value).strip()
    if not token:
        return None
    if len(token) == 4 and token.isdigit():
        return f"{token[:2]}:{token[2:]}:00"
    if len(token) == 5 and ":" in token:
        return f"{token}:00"
    return token


def _source_day_dir(source_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    ts = _normalise_trade_date(trade_date)
    return Path(source_root) / ts.strftime("%Y") / ts.strftime("%m") / ts.strftime("%d")


def _cache_day_dir(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    ts = _normalise_trade_date(trade_date)
    return Path(cache_root) / ts.strftime("%Y") / ts.strftime("%m") / ts.strftime("%d")


def get_cache_data_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_CACHE_FILENAME


def get_legacy_cache_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / LEGACY_CACHE_FILENAME


def get_cache_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    cache_file = get_cache_data_file_path(cache_root, trade_date)
    if cache_file.exists():
        return cache_file

    legacy_file = get_legacy_cache_file_path(cache_root, trade_date)
    if legacy_file.exists():
        return legacy_file
    return cache_file


def get_manifest_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_MANIFEST_FILENAME


def get_stocks_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_STOCKS_FILENAME


def get_minutes_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_MINUTES_FILENAME


def get_fields_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_FIELDS_FILENAME


def get_requested_fields_file_path(cache_root: str | Path, trade_date: str | pd.Timestamp) -> Path:
    return _cache_day_dir(cache_root, trade_date) / DEFAULT_REQUESTED_FIELDS_FILENAME


def _discover_minute_files(
    day_dir: Path,
    start_time: str | None = None,
    end_time: str | None = None,
) -> List[Path]:
    if not day_dir.exists():
        return []

    start_token = _normalise_time_token(start_time)
    end_token = _normalise_time_token(end_time)

    minute_files = sorted(path for path in day_dir.iterdir() if path.suffix.lower() == ".hdf")
    selected: List[Path] = []
    for path in minute_files:
        minute_label = path.stem
        if start_token and minute_label < start_token:
            continue
        if end_token and minute_label > end_token:
            continue
        selected.append(path)
    return selected


def _pick_stock_column(frame: pd.DataFrame) -> str:
    for name in STOCK_COLUMN_CANDIDATES:
        if name in frame.columns:
            return name
    raise ValueError(f"Cannot determine stock column from columns: {list(frame.columns)}")


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _safe_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def normalise_stock_code(value: object) -> str:
    token = str(value).strip().upper()
    if not token or token == "NAN":
        return ""

    match = re.match(r"^(\d{6})[\-\._]?(SZ|SH|BJ)(?:[\-\._]?STOCK)?$", token)
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    match = re.match(r"^(\d{6})\.(SZ|SH|BJ)$", token)
    if match:
        return token

    match = re.match(r"^(SZ|SH|BJ)(\d{6})$", token)
    if match:
        return f"{match.group(2)}.{match.group(1)}"

    match = re.match(r"^(\d{6})$", token)
    if match:
        return token

    return token


def _normalise_stock_whitelist(stock_whitelist: Sequence[str] | None) -> set[str] | None:
    if not stock_whitelist:
        return None
    normalised = {normalise_stock_code(item) for item in stock_whitelist}
    normalised.discard("")
    return normalised or None


def _read_minute_file(
    path: Path,
    requested_fields: Sequence[str],
    stock_whitelist: set[str] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    raw = pd.read_hdf(path)
    stock_column = _pick_stock_column(raw)
    available_fields = [field for field in requested_fields if field in raw.columns]
    if not available_fields:
        return pd.DataFrame(index=pd.Index([], dtype=str)), []

    frame = raw[[stock_column, *available_fields]].copy()
    frame[stock_column] = frame[stock_column].map(normalise_stock_code)
    if stock_whitelist is not None:
        frame = frame[frame[stock_column].isin(stock_whitelist)]
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], dtype=str)), available_fields
    frame = _safe_numeric(frame, available_fields)
    frame = frame.groupby(stock_column, sort=True).last()
    frame.index = frame.index.astype(str)
    return frame.sort_index(), available_fields


def _normalise_cache_format(cache_format: str | None) -> str:
    token = str(cache_format or DEFAULT_CACHE_FORMAT).strip().lower()
    if token not in {"npy", "npz"}:
        raise ValueError(f"Unsupported minute cache format: {cache_format}")
    return token


def _build_entry_from_manifest(manifest: dict[str, Any], manifest_file: Path) -> MinuteTensorCacheEntry:
    cache_file = str(manifest.get("cache_file") or "")
    cache_format = str(manifest.get("cache_format") or "").strip().lower()
    if not cache_format:
        cache_format = "npz" if cache_file.endswith(".npz") else DEFAULT_CACHE_FORMAT

    return MinuteTensorCacheEntry(
        trade_date=str(manifest["trade_date"]),
        source_dir=str(manifest["source_dir"]),
        cache_file=cache_file,
        manifest_file=str(manifest_file),
        dtype=str(manifest["dtype"]),
        requested_fields=[str(item) for item in manifest.get("requested_fields", [])],
        available_fields=[str(item) for item in manifest.get("available_fields", [])],
        missing_fields=[str(item) for item in manifest.get("missing_fields", [])],
        stock_count=int(manifest.get("stock_count", 0)),
        minute_count=int(manifest.get("minute_count", 0)),
        cache_format=cache_format,
        data_file=str(manifest.get("data_file") or cache_file),
        stocks_file=str(manifest.get("stocks_file") or ""),
        minutes_file=str(manifest.get("minutes_file") or ""),
        fields_file=str(manifest.get("fields_file") or ""),
        requested_fields_file=str(manifest.get("requested_fields_file") or ""),
    )


def _load_manifest_entry(cache_root: str | Path, trade_date: str | pd.Timestamp) -> MinuteTensorCacheEntry | None:
    manifest_file = get_manifest_file_path(cache_root, trade_date)
    if not manifest_file.exists():
        return None
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return _build_entry_from_manifest(manifest, manifest_file)


def _entry_files_exist(entry: MinuteTensorCacheEntry) -> bool:
    manifest_path = Path(entry.manifest_file)
    if not manifest_path.exists():
        return False

    if entry.cache_format == "npz":
        return Path(entry.cache_file).exists()

    required = [
        entry.data_file or entry.cache_file,
        entry.stocks_file,
        entry.minutes_file,
        entry.fields_file,
        entry.requested_fields_file,
    ]
    return all(path and Path(path).exists() for path in required)


def _infer_entry_without_manifest(cache_root: str | Path, trade_date: str | pd.Timestamp) -> MinuteTensorCacheEntry:
    manifest_file = get_manifest_file_path(cache_root, trade_date)
    data_file = get_cache_data_file_path(cache_root, trade_date)
    legacy_file = get_legacy_cache_file_path(cache_root, trade_date)
    if data_file.exists():
        return MinuteTensorCacheEntry(
            trade_date=str(_normalise_trade_date(trade_date).date()),
            source_dir=str(_source_day_dir(DEFAULT_SOURCE_ROOT, trade_date)),
            cache_file=str(data_file),
            manifest_file=str(manifest_file),
            dtype="float32",
            requested_fields=[],
            available_fields=[],
            missing_fields=[],
            stock_count=0,
            minute_count=0,
            cache_format="npy",
            data_file=str(data_file),
            stocks_file=str(get_stocks_file_path(cache_root, trade_date)),
            minutes_file=str(get_minutes_file_path(cache_root, trade_date)),
            fields_file=str(get_fields_file_path(cache_root, trade_date)),
            requested_fields_file=str(get_requested_fields_file_path(cache_root, trade_date)),
        )
    if legacy_file.exists():
        return MinuteTensorCacheEntry(
            trade_date=str(_normalise_trade_date(trade_date).date()),
            source_dir=str(_source_day_dir(DEFAULT_SOURCE_ROOT, trade_date)),
            cache_file=str(legacy_file),
            manifest_file=str(manifest_file),
            dtype="float32",
            requested_fields=[],
            available_fields=[],
            missing_fields=[],
            stock_count=0,
            minute_count=0,
            cache_format="npz",
            data_file=str(legacy_file),
        )
    raise FileNotFoundError(f"Minute cache file not found for {trade_date}: {get_cache_file_path(cache_root, trade_date)}")


def _resolve_entry_path(path_value: str, fallback_path: Path) -> Path:
    """Resolve stale manifest paths by preferring existing file then fallback path."""
    if path_value:
        candidate = Path(path_value)
        if candidate.exists():
            return candidate
    return fallback_path


def load_minute_day_cache_entry(cache_root: str | Path, trade_date: str | pd.Timestamp) -> MinuteTensorCacheEntry:
    entry = _load_manifest_entry(cache_root, trade_date)
    if entry is not None:
        return entry
    return _infer_entry_without_manifest(cache_root, trade_date)


def build_minute_day_cache(
    source_root: str | Path,
    cache_root: str | Path,
    trade_date: str | pd.Timestamp,
    fields: Sequence[str],
    start_time: str | None = None,
    end_time: str | None = None,
    stock_whitelist: Sequence[str] | None = None,
    dtype: str = "float32",
    cache_format: str = DEFAULT_CACHE_FORMAT,
    overwrite: bool = False,
) -> MinuteTensorCacheEntry:
    requested_fields = _unique_preserve_order(fields)
    if not requested_fields:
        raise ValueError("No minute fields were provided")
    cache_format = _normalise_cache_format(cache_format)
    normalised_stock_whitelist = _normalise_stock_whitelist(stock_whitelist)

    source_dir = _source_day_dir(source_root, trade_date)
    if not source_dir.exists():
        raise FileNotFoundError(f"Minute source directory not found: {source_dir}")

    cache_file = get_cache_data_file_path(cache_root, trade_date) if cache_format == "npy" else get_legacy_cache_file_path(cache_root, trade_date)
    manifest_file = get_manifest_file_path(cache_root, trade_date)
    if not overwrite:
        existing_entry = _load_manifest_entry(cache_root, trade_date)
        if existing_entry is not None and _entry_files_exist(existing_entry):
            return existing_entry
        if cache_file.exists() and not manifest_file.exists():
            return _infer_entry_without_manifest(cache_root, trade_date)

    minute_files = _discover_minute_files(source_dir, start_time=start_time, end_time=end_time)
    if not minute_files:
        raise ValueError(f"No minute HDF files found under {source_dir}")

    minute_frames: List[tuple[str, pd.DataFrame]] = []
    stock_codes = set()
    available_seen = set()
    for minute_file in minute_files:
        minute_frame, available_fields = _read_minute_file(
            minute_file,
            requested_fields,
            stock_whitelist=normalised_stock_whitelist,
        )
        minute_frames.append((minute_file.stem, minute_frame))
        stock_codes.update(minute_frame.index.tolist())
        available_seen.update(available_fields)

    available_fields = [field for field in requested_fields if field in available_seen]
    missing_fields = [field for field in requested_fields if field not in available_seen]
    if not available_fields:
        raise ValueError(f"None of the requested minute fields were found for {trade_date}: {requested_fields}")
    if not stock_codes:
        raise ValueError(f"No stocks remain after whitelist filtering for {trade_date}")

    ordered_stocks = sorted(stock_codes)
    ordered_minutes = [minute_label for minute_label, _ in minute_frames]
    stock_to_pos = {code: index for index, code in enumerate(ordered_stocks)}

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    dtype_obj = np.dtype(dtype)
    data_shape = (len(ordered_stocks), len(ordered_minutes), len(available_fields))

    if overwrite:
        for stale_path in (
            get_cache_data_file_path(cache_root, trade_date),
            get_legacy_cache_file_path(cache_root, trade_date),
            get_stocks_file_path(cache_root, trade_date),
            get_minutes_file_path(cache_root, trade_date),
            get_fields_file_path(cache_root, trade_date),
            get_requested_fields_file_path(cache_root, trade_date),
        ):
            if stale_path.exists():
                stale_path.unlink()

    if cache_format == "npy":
        data = open_memmap(cache_file, mode="w+", dtype=dtype_obj, shape=data_shape)
        data[:] = np.nan
    else:
        data = np.full(data_shape, np.nan, dtype=dtype_obj)

    for minute_pos, (_, minute_frame) in enumerate(minute_frames):
        if minute_frame.empty:
            continue
        aligned = minute_frame.reindex(columns=available_fields)
        row_positions = np.array([stock_to_pos[code] for code in aligned.index], dtype=np.int64)
        data[row_positions, minute_pos, :] = aligned.to_numpy(dtype=dtype_obj, copy=False)

    stocks_file = ""
    minutes_file = ""
    fields_file = ""
    requested_fields_file = ""
    if cache_format == "npy":
        if hasattr(data, "flush"):
            data.flush()
        stocks_file = str(get_stocks_file_path(cache_root, trade_date))
        minutes_file = str(get_minutes_file_path(cache_root, trade_date))
        fields_file = str(get_fields_file_path(cache_root, trade_date))
        requested_fields_file = str(get_requested_fields_file_path(cache_root, trade_date))
        np.save(stocks_file, np.asarray(ordered_stocks, dtype=np.str_))
        np.save(minutes_file, np.asarray(ordered_minutes, dtype=np.str_))
        np.save(fields_file, np.asarray(available_fields, dtype=np.str_))
        np.save(requested_fields_file, np.asarray(requested_fields, dtype=np.str_))
    else:
        np.savez(
            cache_file,
            data=data,
            stocks=np.asarray(ordered_stocks),
            minutes=np.asarray(ordered_minutes),
            fields=np.asarray(available_fields),
            requested_fields=np.asarray(requested_fields),
            trade_date=np.asarray(str(_normalise_trade_date(trade_date).date())),
        )

    entry = MinuteTensorCacheEntry(
        trade_date=str(_normalise_trade_date(trade_date).date()),
        source_dir=str(source_dir),
        cache_file=str(cache_file),
        manifest_file=str(manifest_file),
        dtype=str(dtype_obj),
        requested_fields=requested_fields,
        available_fields=available_fields,
        missing_fields=missing_fields,
        stock_count=len(ordered_stocks),
        minute_count=len(ordered_minutes),
        cache_format=cache_format,
        data_file=str(cache_file),
        stocks_file=stocks_file,
        minutes_file=minutes_file,
        fields_file=fields_file,
        requested_fields_file=requested_fields_file,
        stock_filter_count=len(normalised_stock_whitelist or ()),
    )
    manifest_file.write_text(json.dumps(asdict(entry), ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def build_minute_cache_range(
    source_root: str | Path,
    cache_root: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    fields: Sequence[str],
    start_time: str | None = None,
    end_time: str | None = None,
    stock_whitelist: Sequence[str] | None = None,
    dtype: str = "float32",
    cache_format: str = DEFAULT_CACHE_FORMAT,
    overwrite: bool = False,
) -> dict[str, object]:
    requested_fields = _unique_preserve_order(fields)
    if not requested_fields:
        raise ValueError("No minute fields were provided")
    cache_format = _normalise_cache_format(cache_format)
    normalised_stock_whitelist = _normalise_stock_whitelist(stock_whitelist)

    start_ts = _normalise_trade_date(start_date)
    end_ts = _normalise_trade_date(end_date)
    if start_ts > end_ts:
        raise ValueError("start_date must be <= end_date")

    processed: List[dict[str, object]] = []
    skipped_missing_source: List[str] = []
    failures: List[dict[str, str]] = []

    for trade_date in pd.date_range(start_ts, end_ts, freq="D"):
        source_dir = _source_day_dir(source_root, trade_date)
        if not source_dir.exists():
            skipped_missing_source.append(str(trade_date.date()))
            continue

        try:
            entry = build_minute_day_cache(
                source_root=source_root,
                cache_root=cache_root,
                trade_date=trade_date,
                fields=requested_fields,
                start_time=start_time,
                end_time=end_time,
                stock_whitelist=normalised_stock_whitelist,
                dtype=dtype,
                cache_format=cache_format,
                overwrite=overwrite,
            )
            processed.append(asdict(entry))
        except Exception as exc:
            failures.append({"trade_date": str(trade_date.date()), "error": str(exc)})

    return {
        "source_root": str(source_root),
        "cache_root": str(cache_root),
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "fields": requested_fields,
        "stock_filter_count": len(normalised_stock_whitelist or ()),
        "dtype": str(np.dtype(dtype)),
        "cache_format": cache_format,
        "processed_count": len(processed),
        "skipped_missing_source_count": len(skipped_missing_source),
        "failure_count": len(failures),
        "processed": processed,
        "skipped_missing_source": skipped_missing_source,
        "failures": failures,
    }


def load_minute_day_cache_axes(cache_root: str | Path, trade_date: str | pd.Timestamp) -> dict[str, np.ndarray]:
    entry = load_minute_day_cache_entry(cache_root, trade_date)
    trade_date_array = np.asarray(str(entry.trade_date))

    if entry.cache_format == "npz":
        cache_file = _resolve_entry_path(entry.cache_file, get_legacy_cache_file_path(cache_root, trade_date))
        with np.load(cache_file, allow_pickle=False) as handle:
            return {
                "stocks": handle["stocks"],
                "minutes": handle["minutes"],
                "fields": handle["fields"],
                "requested_fields": handle["requested_fields"],
                "trade_date": handle.get("trade_date", trade_date_array),
            }

    stocks_file = _resolve_entry_path(entry.stocks_file, get_stocks_file_path(cache_root, trade_date))
    minutes_file = _resolve_entry_path(entry.minutes_file, get_minutes_file_path(cache_root, trade_date))
    fields_file = _resolve_entry_path(entry.fields_file, get_fields_file_path(cache_root, trade_date))
    requested_fields_file = _resolve_entry_path(
        entry.requested_fields_file,
        get_requested_fields_file_path(cache_root, trade_date),
    )

    return {
        "stocks": np.load(stocks_file, allow_pickle=False),
        "minutes": np.load(minutes_file, allow_pickle=False),
        "fields": np.load(fields_file, allow_pickle=False),
        "requested_fields": np.load(requested_fields_file, allow_pickle=False),
        "trade_date": trade_date_array,
    }


def load_minute_day_cache(
    cache_root: str | Path,
    trade_date: str | pd.Timestamp,
    mmap_mode: str | None = None,
) -> dict[str, np.ndarray]:
    entry = load_minute_day_cache_entry(cache_root, trade_date)
    axes = load_minute_day_cache_axes(cache_root, trade_date)

    if entry.cache_format == "npz":
        cache_file = _resolve_entry_path(entry.cache_file, get_legacy_cache_file_path(cache_root, trade_date))
        with np.load(cache_file, allow_pickle=False) as handle:
            payload = {name: handle[name] for name in handle.files}
        payload.setdefault("trade_date", axes["trade_date"])
        return payload

    payload = dict(axes)
    data_file = _resolve_entry_path(entry.data_file or entry.cache_file, get_cache_data_file_path(cache_root, trade_date))
    payload["data"] = np.load(data_file, allow_pickle=False, mmap_mode=mmap_mode)
    return payload