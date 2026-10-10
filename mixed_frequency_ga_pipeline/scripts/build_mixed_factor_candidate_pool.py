#!/usr/bin/env python3
"""Build a diversified mixed-frequency factor candidate pool.

This script reads generated_factors*.txt from selected GA output directories,
joins with existing IC validation results when available, and exports:
1) Candidate CSV with metadata and ranking fields.
2) Candidate TXT in FactorXX format for eval_high_score_factors_ic.py.
3) Summary JSON for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


FACTOR_LINE_RE = re.compile(
    r"^Factor(?P<idx>\d+):\s*(?P<expr>.+?)\s*#\s*score=(?P<score>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FUNC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed-frequency factor candidate pool")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="GA output directory containing generated_factors*.txt (repeatable)",
    )
    parser.add_argument(
        "--ic-csv",
        default="/home/guojh/workspace/gplearn (2)/reports/factor_ic_validation/ic_top5_20260417_175806.csv",
        help="Existing IC ranking CSV to enrich candidates",
    )
    parser.add_argument("--target-size", type=int, default=20, help="Target pool size")
    parser.add_argument("--min-mixed", type=int, default=12, help="Minimum mixed-frequency factors")
    parser.add_argument("--min-decay", type=float, default=0.55, help="Minimum holdout/dev decay ratio")
    parser.add_argument("--max-depth", type=int, default=6, help="Soft max expression depth")
    parser.add_argument(
        "--output-dir",
        default="/home/guojh/workspace/gplearn (2)/reports/factor_ic_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=8,
        help="Maximum selected factors for each major group",
    )
    parser.add_argument(
        "--frequency-allowlist",
        default="",
        help="Comma-separated frequency types to keep, e.g. pure_minute_derived,mixed_frequency",
    )
    return parser.parse_args()


@dataclass
class FactorRow:
    expression: str
    expression_norm: str
    source_score: float
    source_file: str
    source_line: int
    source_run: str
    frequency_type: str
    group: str
    depth: int
    has_holdout: bool = False
    holdout_mean_ic: float = float("nan")
    dev_mean_ic: float = float("nan")
    decay_ratio: float = float("nan")


def normalize_expression(raw: str) -> str:
    expr = raw.strip()
    if "#" in expr:
        expr = expr.split("#", 1)[0].strip()
    expr = re.sub(r"\b_([A-Za-z][A-Za-z0-9_]*)\s*\(", r"\1(", expr)
    expr = re.sub(r"\s+", "", expr)
    return expr


def expression_depth(expr: str) -> int:
    d = 0
    mx = 0
    for ch in expr:
        if ch == "(":
            d += 1
            if d > mx:
                mx = d
        elif ch == ")":
            d = max(0, d - 1)
    return mx


def parse_factor_file(path: Path) -> List[Tuple[str, float, int]]:
    rows: List[Tuple[str, float, int]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        m = FACTOR_LINE_RE.match(raw.strip())
        if not m:
            continue
        expr = m.group("expr").strip()
        score = float(m.group("score"))
        rows.append((expr, score, line_no))
    return rows


def infer_source_run(path: Path) -> str:
    parts = path.parts
    if "ga_outputs" in parts:
        idx = parts.index("ga_outputs")
        if idx - 1 >= 0:
            return parts[idx - 1]
    return path.parent.name


def split_tokens(expr: str) -> Tuple[set, set]:
    all_tokens = set(TOKEN_RE.findall(expr))
    func_tokens = set(FUNC_RE.findall(expr))
    var_tokens = {t for t in all_tokens if t not in func_tokens}
    minute_vars = {t for t in var_tokens if t.startswith("min_")}
    daily_vars = var_tokens - minute_vars
    return minute_vars, daily_vars


def infer_frequency_type(expr: str) -> str:
    minute_vars, daily_vars = split_tokens(expr)
    if minute_vars and daily_vars:
        return "mixed_frequency"
    if minute_vars and not daily_vars:
        return "pure_minute_derived"
    if daily_vars and not minute_vars:
        return "pure_daily"
    return "unknown"


def infer_group(expr: str) -> str:
    e = expr
    if "d_ratio(" in e:
        return "d_ratio_vol_norm"
    if "d_norm_spread(" in e:
        return "d_norm_spread_fusion"
    if re.search(r"\bday_mean_\d+\(", e):
        return "day_mean_smoothing"
    if any(k in e for k in ("am_beta(", "am_", "pm_", "open15_", "open30_", "close15_", "close30_")):
        return "time_segment_structure"
    if "mean_spread(" in e:
        return "mean_spread_interaction"
    return "other"


def discover_factor_files(source_dirs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for src in source_dirs:
        if not src.exists():
            continue
        latest = src / "generated_factors_latest.txt"
        if latest.exists():
            files.append(latest)
        stamp_files = sorted(src.glob("generated_factors_*.txt"))
        stamp_files = [p for p in stamp_files if p.name != "generated_factors_latest.txt"]
        files.extend(stamp_files[-3:])
    uniq = sorted({str(p): p for p in files}.values(), key=lambda p: str(p))
    return uniq


def load_ic_map(ic_csv: Path) -> Dict[str, Dict[str, float]]:
    if not ic_csv.exists():
        return {}
    df = pd.read_csv(ic_csv)
    required = {"expression", "holdout_mean_ic", "dev_mean_ic"}
    if not required.issubset(set(df.columns)):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for row in df.itertuples(index=False):
        expr = normalize_expression(str(getattr(row, "expression", "")))
        if not expr:
            continue
        holdout = float(getattr(row, "holdout_mean_ic", float("nan")))
        dev = float(getattr(row, "dev_mean_ic", float("nan")))
        decay = float("nan")
        if pd.notna(dev) and abs(dev) > 1e-12 and pd.notna(holdout):
            decay = holdout / dev
        out[expr] = {
            "holdout_mean_ic": holdout,
            "dev_mean_ic": dev,
            "decay_ratio": decay,
        }
    return out


def build_rows(files: Sequence[Path], ic_map: Dict[str, Dict[str, float]]) -> List[FactorRow]:
    best: Dict[str, FactorRow] = {}
    for f in files:
        source_run = infer_source_run(f)
        for expr, score, line_no in parse_factor_file(f):
            expr_norm = normalize_expression(expr)
            if not expr_norm:
                continue
            row = FactorRow(
                expression=expr,
                expression_norm=expr_norm,
                source_score=float(score),
                source_file=str(f),
                source_line=int(line_no),
                source_run=source_run,
                frequency_type=infer_frequency_type(expr_norm),
                group=infer_group(expr_norm),
                depth=expression_depth(expr_norm),
            )
            ic = ic_map.get(expr_norm)
            if ic:
                row.has_holdout = True
                row.holdout_mean_ic = float(ic["holdout_mean_ic"])
                row.dev_mean_ic = float(ic["dev_mean_ic"])
                row.decay_ratio = float(ic["decay_ratio"])

            old = best.get(expr_norm)
            if old is None or row.source_score > old.source_score:
                best[expr_norm] = row

    rows = list(best.values())
    return rows


def rank_key(row: FactorRow, min_decay: float, max_depth: int) -> Tuple:
    has_ic = 1 if row.has_holdout else 0
    pass_decay = 1
    if row.has_holdout and pd.notna(row.decay_ratio):
        pass_decay = 1 if row.decay_ratio >= min_decay else 0
    holdout = row.holdout_mean_ic if row.has_holdout and pd.notna(row.holdout_mean_ic) else -999.0
    depth_penalty = 0 if row.depth <= max_depth else -1
    return (has_ic, pass_decay, holdout, row.source_score, depth_penalty, -row.depth)


def select_candidates(
    rows: Sequence[FactorRow],
    target_size: int,
    min_mixed: int,
    min_decay: float,
    max_depth: int,
    max_per_group: int,
) -> List[FactorRow]:
    by_group: Dict[str, List[FactorRow]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r)

    for g in by_group:
        by_group[g] = sorted(by_group[g], key=lambda x: rank_key(x, min_decay, max_depth), reverse=True)

    group_order = sorted(by_group.keys(), key=lambda g: len(by_group[g]), reverse=True)
    selected: List[FactorRow] = []
    selected_norm = set()
    group_count: Dict[str, int] = {g: 0 for g in group_order}

    # Pass 1: ensure mixed-frequency coverage using round-robin by groups.
    while len(selected) < target_size and len([r for r in selected if r.frequency_type == "mixed_frequency"]) < min_mixed:
        advanced = False
        for g in group_order:
            if group_count[g] >= max_per_group:
                continue
            while by_group[g]:
                cand = by_group[g].pop(0)
                if cand.expression_norm in selected_norm:
                    continue
                if cand.frequency_type != "mixed_frequency":
                    continue
                selected.append(cand)
                selected_norm.add(cand.expression_norm)
                group_count[g] += 1
                advanced = True
                break
        if not advanced:
            break

    # Pass 2: fill remaining slots with best available under group cap.
    all_sorted = sorted(rows, key=lambda x: rank_key(x, min_decay, max_depth), reverse=True)
    for cand in all_sorted:
        if len(selected) >= target_size:
            break
        if cand.expression_norm in selected_norm:
            continue
        if group_count.get(cand.group, 0) >= max_per_group:
            continue
        selected.append(cand)
        selected_norm.add(cand.expression_norm)
        group_count[cand.group] = group_count.get(cand.group, 0) + 1

    return selected


def export_outputs(selected: Sequence[FactorRow], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"candidate_pool_20_{ts}.csv"
    txt_path = output_dir / f"candidate_pool_20_{ts}.txt"
    json_path = output_dir / f"candidate_pool_20_summary_{ts}.json"

    df = pd.DataFrame([asdict(r) for r in selected])
    if not df.empty:
        df = df.sort_values(
            by=["has_holdout", "holdout_mean_ic", "source_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    lines = ["# Candidate factor expressions for one-shot light validation", f"# generated_at: {ts}"]
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            f"Factor{i:02d}: {row.expression}  # score={float(row.source_score):.6f}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "generated_at": ts,
        "count": int(len(df)),
        "mixed_count": int((df["frequency_type"] == "mixed_frequency").sum()) if not df.empty else 0,
        "group_counts": df["group"].value_counts().to_dict() if not df.empty else {},
        "frequency_counts": df["frequency_type"].value_counts().to_dict() if not df.empty else {},
        "outputs": {
            "candidate_csv": str(csv_path),
            "candidate_txt": str(txt_path),
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"candidate_csv": csv_path, "candidate_txt": txt_path, "summary_json": json_path}


def main() -> None:
    args = parse_args()

    source_dirs = [Path(p) for p in args.source_dir]
    if not source_dirs:
        source_dirs = [
            Path("/data/intership/guojh/GA和GRU/tmp/ga_3d_native_zz500_2019plus_neu_all_vwap_5d_expB/ga_outputs"),
            Path("/data/intership/guojh/GA和GRU/tmp/ga_3d_native_zz500_2y_fullfields_neu_all_vwap_5d_expB/ga_outputs"),
        ]

    ic_map = load_ic_map(Path(args.ic_csv))
    files = discover_factor_files(source_dirs)
    rows = build_rows(files, ic_map)

    allow_raw = [s.strip() for s in str(args.frequency_allowlist).split(",") if s.strip()]
    allow = set(allow_raw)
    if allow:
        rows = [r for r in rows if r.frequency_type in allow]

    selected = select_candidates(
        rows=rows,
        target_size=max(1, int(args.target_size)),
        min_mixed=max(0, int(args.min_mixed)),
        min_decay=float(args.min_decay),
        max_depth=max(1, int(args.max_depth)),
        max_per_group=max(1, int(args.max_per_group)),
    )
    outputs = export_outputs(selected, Path(args.output_dir))

    print("=== Mixed Factor Candidate Pool Built ===")
    print(f"source_files: {len(files)}")
    print(f"raw_unique_candidates: {len(rows)}")
    print(f"selected_candidates: {len(selected)}")
    print(f"candidate_csv: {outputs['candidate_csv']}")
    print(f"candidate_txt: {outputs['candidate_txt']}")
    print(f"summary_json: {outputs['summary_json']}")


if __name__ == "__main__":
    main()
