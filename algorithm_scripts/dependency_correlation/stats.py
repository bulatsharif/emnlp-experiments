"""Correlation summaries."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan, math.nan
    result = spearmanr(x, y) if kind == "spearman" else pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def summarize(records: list[dict]) -> list[dict]:
    by_task = defaultdict(list)
    for row in records:
        by_task[row["task"]].append(row)

    out = []
    for task, rows in sorted(by_task.items()):
        y = np.array([row["true_dependency"] for row in rows], dtype=float)
        a = np.array([row.get("attention_score", math.nan) for row in rows], dtype=float)
        h = np.array([row["hidden_cosine"] for row in rows], dtype=float)
        l = np.array([row["logit_cosine"] for row in rows], dtype=float)
        attention_rho, attention_p = _corr(a, y, "spearman")
        hidden_rho, hidden_p = _corr(h, y, "spearman")
        logit_rho, logit_p = _corr(l, y, "spearman")
        attention_pearson, attention_pearson_p = _corr(a, y, "pearson")
        hidden_pearson, hidden_pearson_p = _corr(h, y, "pearson")
        logit_pearson, logit_pearson_p = _corr(l, y, "pearson")
        out.append(
            {
                "task": task,
                "pairs": len(rows),
                "spearman_attention": attention_rho,
                "spearman_logit": logit_rho,
                "spearman_hidden": hidden_rho,
                "p_attention": attention_p,
                "p_logit": logit_p,
                "p_hidden": hidden_p,
                "pearson_attention": attention_pearson,
                "pearson_logit": logit_pearson,
                "pearson_hidden": hidden_pearson,
                "pearson_p_attention": attention_pearson_p,
                "pearson_p_logit": logit_pearson_p,
                "pearson_p_hidden": hidden_pearson_p,
            }
        )
    return out


def print_table(rows: list[dict]) -> None:
    print("\n| Task | Pairs | Spearman attn | Spearman logit | Spearman hidden |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['task']} | {row['pairs']} | "
            f"{row['spearman_attention']:.4f} | {row['spearman_logit']:.4f} | "
            f"{row['spearman_hidden']:.4f} |"
        )
