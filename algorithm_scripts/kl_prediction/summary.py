"""Small reporting helpers for pairwise probe records."""

from __future__ import annotations

import math

import numpy as np


def finite(records: list[dict]) -> list[dict]:
    return [
        row
        for row in records
        if math.isfinite(row["cosine"]) and math.isfinite(row["kl"])
    ]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty_like(order, dtype=float)
    out[order] = np.arange(len(values), dtype=float)
    return out


def summarize(records: list[dict], bins: int) -> dict:
    rows = finite(records)
    cos = np.array([row["cosine"] for row in rows], dtype=float)
    kl = np.array([row["kl"] for row in rows], dtype=float)
    summary = {
        "total": len(records),
        "finite": len(rows),
        "pearson": _corr(cos, kl),
        "spearman": _corr(_ranks(cos), _ranks(kl)),
        "cos_mean": float(cos.mean()) if len(cos) else math.nan,
        "cos_min": float(cos.min()) if len(cos) else math.nan,
        "cos_max": float(cos.max()) if len(cos) else math.nan,
        "kl_mean": float(kl.mean()) if len(kl) else math.nan,
        "kl_min": float(kl.min()) if len(kl) else math.nan,
        "kl_max": float(kl.max()) if len(kl) else math.nan,
        "bins": [],
    }
    if not len(rows):
        return summary

    for idx in np.array_split(np.argsort(cos, kind="mergesort"), bins):
        summary["bins"].append(
            {
                "n": int(len(idx)),
                "cos_mean": float(cos[idx].mean()),
                "cos_min": float(cos[idx].min()),
                "cos_max": float(cos[idx].max()),
                "kl_mean": float(kl[idx].mean()),
            }
        )
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== hidden cosine vs conditional KL ===")
    print(f"pairs:    {summary['finite']}/{summary['total']} finite")
    print(f"pearson:  {summary['pearson']:.4f}")
    print(f"spearman: {summary['spearman']:.4f}")
    print(
        f"cosine:   mean={summary['cos_mean']:.6f} "
        f"range=[{summary['cos_min']:.6f}, {summary['cos_max']:.6f}]"
    )
    print(
        f"KL:       mean={summary['kl_mean']:.6f} "
        f"range=[{summary['kl_min']:.6f}, {summary['kl_max']:.6f}]"
    )
    print("\nKL by cosine quantile:")
    print("bin  n    cos_mean   cos_range          kl_mean")
    for i, row in enumerate(summary["bins"]):
        print(
            f"{i:>3}  {row['n']:>4}  {row['cos_mean']:.6f}  "
            f"[{row['cos_min']:.6f}, {row['cos_max']:.6f}]  {row['kl_mean']:.6f}"
        )

