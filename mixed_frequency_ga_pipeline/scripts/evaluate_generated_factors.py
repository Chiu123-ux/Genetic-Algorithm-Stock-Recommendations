#!/usr/bin/env python3
"""Evaluate generated GA factor expressions from a text file.

Expected input format:
    Factor01: expr  # score=0.1234

Outputs:
1) Console summary.
2) Optional JSON report for downstream automation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple


FACTOR_LINE_RE = re.compile(r"^Factor(?P<idx>\d+):\s*(?P<expr>.+?)\s*#\s*score=(?P<score>-?\d+(?:\.\d+)?)\s*$")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class FactorRow:
    index: int
    expression: str
    score: float


def parse_factor_file(file_path: Path) -> List[FactorRow]:
    rows: List[FactorRow] = []
    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = FACTOR_LINE_RE.match(line)
        if not m:
            continue
        rows.append(
            FactorRow(
                index=int(m.group("idx")),
                expression=m.group("expr").strip(),
                score=float(m.group("score")),
            )
        )
    return rows


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def normalize_expression(expr: str) -> str:
    # Replace numeric literals to detect near-duplicate structures.
    out = NUM_RE.sub("NUM", expr)
    out = re.sub(r"\s+", "", out)
    return out


def weak_signature(expr: str) -> str:
    """Token-bag signature for near-duplicate detection.

    This signature intentionally ignores exact structure and focuses on
    operators/variables composition, which is useful for catching expression
    variants that differ by only light wrappers.
    """
    normalized = NUM_RE.sub("NUM", expr)
    tokens = TOKEN_RE.findall(normalized)
    if not tokens:
        return ""
    bucket: Dict[str, int] = {}
    for tok in tokens:
        bucket[tok] = bucket.get(tok, 0) + 1
    parts = [f"{k}:{bucket[k]}" for k in sorted(bucket)]
    return "|".join(parts)


def expression_depth(expr: str) -> int:
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == ")":
            depth = max(depth - 1, 0)
    return max_depth


def build_duplicate_clusters(rows: List[FactorRow]) -> List[List[FactorRow]]:
    grouped: Dict[str, List[FactorRow]] = {}
    for row in rows:
        sig = normalize_expression(row.expression)
        grouped.setdefault(sig, []).append(row)
    clusters = [group for group in grouped.values() if len(group) > 1]
    clusters.sort(key=lambda c: max(r.score for r in c), reverse=True)
    return clusters


def build_near_duplicate_clusters(rows: List[FactorRow]) -> List[List[FactorRow]]:
    grouped: Dict[str, List[FactorRow]] = {}
    for row in rows:
        sig = weak_signature(row.expression)
        grouped.setdefault(sig, []).append(row)
    clusters = [group for group in grouped.values() if len(group) > 1]
    clusters.sort(key=lambda c: max(r.score for r in c), reverse=True)
    return clusters


def select_keep_candidates(rows: List[FactorRow], max_keep: int = 10) -> List[FactorRow]:
    # Keep top factors while avoiding normalized duplicates.
    selected: List[FactorRow] = []
    seen_signatures: set[str] = set()
    for row in sorted(rows, key=lambda r: r.score, reverse=True):
        sig = normalize_expression(row.expression)
        if sig in seen_signatures:
            continue
        selected.append(row)
        seen_signatures.add(sig)
        if len(selected) >= max_keep:
            break
    return selected


def evaluate(rows: List[FactorRow]) -> Dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "message": "No factors parsed from file",
        }

    scores = [r.score for r in rows]
    sorted_scores = sorted(scores)
    non_zero = [s for s in scores if s != 0.0]

    depth_values = [expression_depth(r.expression) for r in rows]

    clusters = build_duplicate_clusters(rows)
    near_clusters = build_near_duplicate_clusters(rows)
    keep = select_keep_candidates(rows, max_keep=10)

    top10 = sorted(rows, key=lambda r: r.score, reverse=True)[:10]

    return {
        "count": len(rows),
        "score_stats": {
            "mean": round(mean(scores), 6),
            "median": round(median(scores), 6),
            "p75": round(percentile(sorted_scores, 75), 6),
            "p90": round(percentile(sorted_scores, 90), 6),
            "max": round(max(scores), 6),
            "min": round(min(scores), 6),
            "non_zero_ratio": round(len(non_zero) / len(scores), 6),
            "zero_ratio": round((len(scores) - len(non_zero)) / len(scores), 6),
        },
        "complexity": {
            "depth_mean": round(mean(depth_values), 4),
            "depth_median": round(median(depth_values), 4),
            "depth_max": max(depth_values),
        },
        "top10": [
            {
                "factor": f"Factor{r.index:02d}",
                "score": round(r.score, 6),
                "expression": r.expression,
            }
            for r in top10
        ],
        "duplicate_clusters": [
            [
                {
                    "factor": f"Factor{r.index:02d}",
                    "score": round(r.score, 6),
                    "expression": r.expression,
                }
                for r in cluster
            ]
            for cluster in clusters
        ],
        "near_duplicate_clusters": [
            [
                {
                    "factor": f"Factor{r.index:02d}",
                    "score": round(r.score, 6),
                    "expression": r.expression,
                }
                for r in cluster
            ]
            for cluster in near_clusters
        ],
        "keep_candidates": [
            {
                "factor": f"Factor{r.index:02d}",
                "score": round(r.score, 6),
                "expression": r.expression,
            }
            for r in keep
        ],
    }


def print_summary(report: Dict[str, object]) -> None:
    if report.get("count", 0) == 0:
        print(report.get("message", "No report"))
        return

    stats = report["score_stats"]
    comp = report["complexity"]
    print("=== Factor Evaluation Summary ===")
    print(f"Count: {report['count']}")
    print(
        "Score stats: "
        f"mean={stats['mean']}, median={stats['median']}, p75={stats['p75']}, "
        f"p90={stats['p90']}, max={stats['max']}, zero_ratio={stats['zero_ratio']}"
    )
    print(
        "Complexity: "
        f"depth_mean={comp['depth_mean']}, depth_median={comp['depth_median']}, depth_max={comp['depth_max']}"
    )

    print("\nTop10 factors:")
    for item in report["top10"]:
        print(f"- {item['factor']} score={item['score']}: {item['expression']}")

    clusters: List[List[Dict[str, object]]] = report["duplicate_clusters"]  # type: ignore[assignment]
    print(f"\nDuplicate clusters (normalized expression): {len(clusters)}")
    for i, cluster in enumerate(clusters[:10], start=1):
        members = ", ".join(f"{x['factor']}({x['score']})" for x in cluster)
        print(f"- Cluster {i}: {members}")

    near_clusters: List[List[Dict[str, object]]] = report["near_duplicate_clusters"]  # type: ignore[assignment]
    print(f"\nNear-duplicate clusters (token-bag): {len(near_clusters)}")
    for i, cluster in enumerate(near_clusters[:10], start=1):
        members = ", ".join(f"{x['factor']}({x['score']})" for x in cluster)
        print(f"- Near cluster {i}: {members}")

    print("\nSuggested keep candidates (deduped):")
    for item in report["keep_candidates"]:
        print(f"- {item['factor']} score={item['score']}: {item['expression']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated GA factors")
    parser.add_argument("--input", required=True, help="Path to generated_factors_*.txt")
    parser.add_argument("--json-output", default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = parse_factor_file(input_path)
    report = evaluate(rows)
    print_summary(report)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report saved to: {out}")


if __name__ == "__main__":
    main()
