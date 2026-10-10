"""High-level orchestration helpers for factor fetching jobs.

此模块封装因子抓取流程的高层调用，输出简洁的任务摘要，供 CLI 或其他流程复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.data_fetch.factor_fetcher import FactorFetchResult, run_factor_fetch
from src.utils.config_loader import get_config_value


@dataclass
class FactorJobSummary:
    """封装一次因子抓取任务的关键产物路径和统计信息。"""
    job: str        
    factor_file: Path
    close_file: Optional[Path]
    latest_factor_file: Path
    latest_close_file: Optional[Path]
    chunk_files: List[Path]
    stock_count: int
    date_range: Tuple[str, str]

def run_factor_job(job_name: str, output_directory: str | None = None) -> FactorJobSummary:
    """Run a configured factor fetch job and return a concise summary."""

    # 调用底层抓取逻辑，按照配置下载并整理因子数据
    result: FactorFetchResult = run_factor_fetch(job_name, output_directory)

    # 将底层输出映射为较小的结构，便于 CLI 打印或记录
    summary = FactorJobSummary(
        job=job_name,
        factor_file=result.factor_file,
        close_file=result.close_file,
        latest_factor_file=result.latest_factor_file,
        latest_close_file=result.latest_close_file,
        chunk_files=result.chunk_files,
        stock_count=result.metadata.get("stock_count", len(result.dataset.stocks)),
        date_range=tuple(result.metadata.get("date_range", ("", ""))),
    )
    return summary


def latest_factor_paths(job_name: str) -> dict[str, str]:
    """Return expected factor file names for the configured job without running it."""

    # 只利用配置生成预期输出路径，避免重复抓取
    config = get_config_value("factor_fetch", job_name)
    output_dir = get_config_value("paths", "output_directory", default="/home/intership/guojh/GA和GRU")
    prefix = config["output_prefix"]
    suffix = f"{config['max_stocks']}stocks_{config['years']}years"
    return {
        "factors": f"{output_dir}/{prefix}_{suffix}_factors.pkl",
        "close": f"{output_dir}/{prefix}_{suffix}_close.pkl",
    }
