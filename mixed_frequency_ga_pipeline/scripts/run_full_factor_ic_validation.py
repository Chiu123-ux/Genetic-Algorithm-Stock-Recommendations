#!/usr/bin/env python3
"""Orchestrate full factor discovery and IC validation workflow.

Workflow:
1) Discover all generated_factors*.txt files from GA output roots.
2) Persist an inventory report (csv/json).
3) Run batch IC validation using eval_high_score_factors_ic.py.
4) Build high-IC whitelist and blacklist artifacts from the latest detail output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config_3d_native_neu_all_vwap_5d_expB_zz500_2019plus.json"
DEFAULT_DISCOVERY_ROOT = Path("/data/intership/guojh/GA和GRU/tmp")
DEFAULT_OUTPUT_DIR = Path("/data/intership/guojh/GA和GRU/reports/factor_ic_validation")


@dataclass
class FactorFileRecord:
    file_path: str
    file_type: str
    experiment_id: str
    source: str
    exists: bool
    size_bytes: int
    mtime: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full factor IC validation workflow")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Unified config json path")
    parser.add_argument(
        "--discovery-root",
        action="append",
        default=[],
        help="Directory root to discover generated_factors*.txt recursively (repeatable)",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(PROJECT_ROOT),
        help="Workspace root for scanning config references",
    )
    parser.add_argument("--log-file", default="", help="Optional GA log path for extra candidates")
    parser.add_argument("--top-n", type=int, default=300, help="Top-N deduped candidates to evaluate")
    parser.add_argument("--lookback-years", type=float, default=2.0, help="Lookback years")
    parser.add_argument("--holdout-months", type=int, default=6, help="Holdout months")
    parser.add_argument("--min-valid-days", type=int, default=20, help="Min valid holdout days")
    parser.add_argument("--cutoff-date", default=None, help="Optional cutoff date YYYY-MM-DD")
    parser.add_argument("--factors-glob", default="**/ga_outputs/generated_factors*.txt", help="Glob for factors scan")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--whitelist-top-n", type=int, default=50, help="Top-N for whitelist export")
    return parser.parse_args()


def _infer_file_type(path: Path) -> str:
    name = path.name
    if name == "generated_factors_latest.txt":
        return "latest"
    if name == "generated_factors_progress.txt":
        return "progress"
    if re.match(r"generated_factors_\d{8}_\d{6}\.txt", name):
        return "timestamp"
    return "other"


def _infer_experiment_id(path: Path) -> str:
    # .../<experiment>/ga_outputs/generated_factors*.txt
    parts = path.parts
    if "ga_outputs" in parts:
        idx = parts.index("ga_outputs")
        if idx - 1 >= 0:
            return parts[idx - 1]
    return "unknown"


def discover_from_root(root: Path, glob_pattern: str) -> List[FactorFileRecord]:
    records: List[FactorFileRecord] = []
    if not root.exists():
        return records

    for p in sorted(root.glob(glob_pattern)):
        if not p.is_file():
            continue
        stat = p.stat()
        records.append(
            FactorFileRecord(
                file_path=str(p),
                file_type=_infer_file_type(p),
                experiment_id=_infer_experiment_id(p),
                source="filesystem",
                exists=True,
                size_bytes=int(stat.st_size),
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            )
        )
    return records


def discover_from_configs(workspace_root: Path) -> List[FactorFileRecord]:
    records: List[FactorFileRecord] = []
    for cfg_path in sorted(workspace_root.glob("*.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        ga_output = (cfg.get("ga_pipeline") or {}).get("output_directory")
        factor_expression = (cfg.get("gru_pipeline") or {}).get("factor_expression_file")

        candidates: List[Path] = []
        if isinstance(ga_output, str) and ga_output.strip():
            candidates.append(Path(ga_output) / "generated_factors_latest.txt")
        if isinstance(factor_expression, str) and factor_expression.strip():
            candidates.append(Path(factor_expression))

        for c in candidates:
            exists = c.exists()
            size_bytes = int(c.stat().st_size) if exists else 0
            mtime = (
                datetime.fromtimestamp(c.stat().st_mtime).isoformat(timespec="seconds")
                if exists
                else ""
            )
            records.append(
                FactorFileRecord(
                    file_path=str(c),
                    file_type=_infer_file_type(c),
                    experiment_id=_infer_experiment_id(c),
                    source=f"config:{cfg_path.name}",
                    exists=bool(exists),
                    size_bytes=size_bytes,
                    mtime=mtime,
                )
            )
    return records


def dedupe_records(records: Sequence[FactorFileRecord]) -> List[FactorFileRecord]:
    best: Dict[str, FactorFileRecord] = {}
    priority = {
        "filesystem": 3,
    }

    for r in records:
        old = best.get(r.file_path)
        new_p = priority.get(r.source, 1)
        old_p = priority.get(old.source, 1) if old else -1
        if old is None or (new_p > old_p) or (r.exists and not old.exists):
            best[r.file_path] = r
    return sorted(best.values(), key=lambda x: (x.file_path, x.source))


def write_inventory(records: Sequence[FactorFileRecord], output_dir: Path, ts: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"factor_file_inventory_{ts}.csv"
    json_path = output_dir / f"factor_file_inventory_{ts}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "file_type",
                "experiment_id",
                "source",
                "exists",
                "size_bytes",
                "mtime",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    payload = {
        "generated_at": ts,
        "total_rows": len(records),
        "exists_rows": int(sum(1 for r in records if r.exists)),
        "records": [asdict(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"inventory_csv": csv_path, "inventory_json": json_path}


def run_batch_ic_eval(
    config_path: Path,
    factors_dir: Path,
    factors_glob: str,
    output_dir: Path,
    top_n: int,
    lookback_years: float,
    holdout_months: int,
    min_valid_days: int,
    cutoff_date: str | None,
    log_file: str,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_high_score_factors_ic.py"),
        "--config",
        str(config_path),
        "--factors-dir",
        str(factors_dir),
        "--factors-glob",
        factors_glob,
        "--top-n",
        str(top_n),
        "--lookback-years",
        str(lookback_years),
        "--holdout-months",
        str(holdout_months),
        "--min-valid-days",
        str(min_valid_days),
        "--output-dir",
        str(output_dir),
    ]

    if log_file.strip():
        cmd += ["--log-file", log_file.strip()]
    if cutoff_date:
        cmd += ["--cutoff-date", cutoff_date]

    subprocess.run(cmd, check=True)


def _latest_file(output_dir: Path, glob_pattern: str) -> Path:
    files = sorted(output_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {glob_pattern}")
    return files[-1]


def build_whitelist_blacklist(output_dir: Path, whitelist_top_n: int, ts: str) -> Dict[str, Path]:
    detail_csv = _latest_file(output_dir, "high_score_factor_ic_detail_*.csv")
    df = pd.read_csv(detail_csv)

    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    failed = df[df["status"] != "ok"].copy() if "status" in df.columns else pd.DataFrame()

    if not ok.empty:
        ok["holdout_mean_ic"] = pd.to_numeric(ok["holdout_mean_ic"], errors="coerce")
        ok = ok.sort_values(by=["holdout_mean_ic", "holdout_icir", "dev_mean_ic"], ascending=[False, False, False])

    top = ok.head(max(1, int(whitelist_top_n))) if not ok.empty else pd.DataFrame()

    whitelist_txt = output_dir / f"high_ic_whitelist_top{whitelist_top_n}_{ts}.txt"
    whitelist_csv = output_dir / f"high_ic_whitelist_top{whitelist_top_n}_{ts}.csv"
    blacklist_csv = output_dir / f"high_ic_blacklist_failed_{ts}.csv"

    lines: List[str] = [f"# generated_at={ts}", f"# source_detail_csv={detail_csv}"]
    if top.empty:
        lines.append("# no_ok_factor")
    else:
        for i, row in enumerate(top.itertuples(index=False), start=1):
            expr = str(getattr(row, "expression", "")).strip()
            hold_ic = float(getattr(row, "holdout_mean_ic", 0.0))
            hold_icir = float(getattr(row, "holdout_icir", 0.0))
            lines.append(f"Factor{i:02d}: {expr}  # holdout_mean_ic={hold_ic:.6f}, holdout_icir={hold_icir:.6f}")

    whitelist_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    top.to_csv(whitelist_csv, index=False)
    failed.to_csv(blacklist_csv, index=False)

    return {
        "source_detail_csv": detail_csv,
        "whitelist_txt": whitelist_txt,
        "whitelist_csv": whitelist_csv,
        "blacklist_csv": blacklist_csv,
    }


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_path = Path(args.config).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    discovery_roots = [Path(p).resolve() for p in args.discovery_root]
    if not discovery_roots:
        discovery_roots = [DEFAULT_DISCOVERY_ROOT]

    records: List[FactorFileRecord] = []
    for root in discovery_roots:
        records.extend(discover_from_root(root, args.factors_glob))
    records.extend(discover_from_configs(workspace_root))

    deduped = dedupe_records(records)
    inventory_paths = write_inventory(deduped, output_dir, ts)

    run_batch_ic_eval(
        config_path=config_path,
        factors_dir=discovery_roots[0],
        factors_glob=args.factors_glob,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n)),
        lookback_years=float(args.lookback_years),
        holdout_months=max(1, int(args.holdout_months)),
        min_valid_days=max(1, int(args.min_valid_days)),
        cutoff_date=args.cutoff_date,
        log_file=args.log_file,
    )

    list_paths = build_whitelist_blacklist(output_dir, args.whitelist_top_n, ts)

    summary = {
        "timestamp": ts,
        "config": str(config_path),
        "discovery_roots": [str(p) for p in discovery_roots],
        "factors_glob": args.factors_glob,
        "top_n": int(args.top_n),
        "holdout_months": int(args.holdout_months),
        "lookback_years": float(args.lookback_years),
        "inventory": {k: str(v) for k, v in inventory_paths.items()},
        "outputs": {k: str(v) for k, v in list_paths.items()},
    }
    summary_path = output_dir / f"full_factor_ic_run_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Full Factor IC Validation Run ===")
    print(f"summary: {summary_path}")
    print(f"inventory_csv: {inventory_paths['inventory_csv']}")
    print(f"whitelist_txt: {list_paths['whitelist_txt']}")
    print(f"blacklist_csv: {list_paths['blacklist_csv']}")


if __name__ == "__main__":
    main()
