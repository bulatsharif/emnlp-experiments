"""KL predictor training and reporting."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def print_collection_summary(records: list[dict]) -> None:
    rows = [row for row in records if math.isfinite(row["kl"])]
    kl = torch.tensor([row["kl"] for row in rows], dtype=torch.float64)

    print("\n=== pair KL collection ===")
    print(f"pairs: {len(rows)}/{len(records)} finite KL")
    if len(rows) == 0:
        return
    print(
        f"KL:    mean={kl.mean().item():.6f} "
        f"range=[{kl.min().item():.6f}, {kl.max().item():.6f}]"
    )


def _regression_metrics(pred: torch.Tensor, target: torch.Tensor, baseline: float) -> dict:
    if len(target) == 0:
        return {
            "rmse": math.nan,
            "mae": math.nan,
            "r2": math.nan,
            "baseline_rmse": math.nan,
        }

    residual = pred - target
    baseline_residual = target - baseline
    sse = torch.sum(residual.square())
    sst = torch.sum((target - target.mean()).square())
    r2 = math.nan if sst.item() == 0.0 else float(1.0 - sse / sst)
    return {
        "rmse": float(torch.sqrt(torch.mean(residual.square()))),
        "mae": float(torch.mean(torch.abs(residual))),
        "r2": r2,
        "baseline_rmse": float(torch.sqrt(torch.mean(baseline_residual.square()))),
    }


def _feature_split(
    features: list[torch.Tensor],
    targets: list[float],
    train_fraction: float,
    seed: int,
) -> dict:
    if not features:
        return {"ok": False, "reason": "no pair features were collected"}

    x = torch.stack(features).to(dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    finite = torch.isfinite(y) & torch.isfinite(x).all(dim=1)
    x = x[finite]
    y = y[finite]
    if len(y) < 2:
        return {"ok": False, "reason": "need at least two finite KL targets"}

    train_fraction = min(max(train_fraction, 0.0), 1.0)
    train_n = int(round(len(y) * train_fraction))
    train_n = min(max(train_n, 1), len(y) - 1)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=generator)
    return {
        "ok": True,
        "x": x,
        "x_train": x[perm[:train_n]],
        "x_test": x[perm[train_n:]],
        "y_train": y[perm[:train_n]],
        "y_test": y[perm[train_n:]],
    }


def linear_kl_probe(
    features: list[torch.Tensor],
    targets: list[float],
    train_fraction: float,
    ridge: float,
    seed: int,
) -> dict:
    split = _feature_split(features, targets, train_fraction, seed)
    if not split["ok"]:
        return split

    x = split["x"]
    x_train = split["x_train"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    y_mean = y_train.mean()
    ridge = max(float(ridge), 0.0)
    gram = x_train @ x_train.T
    if ridge > 0.0:
        gram = gram + ridge * torch.eye(len(x_train), dtype=gram.dtype)

    try:
        alpha = torch.linalg.solve(gram, y_train - y_mean)
    except RuntimeError:
        alpha = torch.linalg.lstsq(gram, (y_train - y_mean)[:, None]).solution[:, 0]

    weights = x_train.T @ alpha
    baseline = float(y_mean)
    return {
        "ok": True,
        "pairs": int(len(y_train) + len(y_test)),
        "hidden_dim": int(x.shape[1] // 2),
        "feature_dim": int(x.shape[1]),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "ridge": ridge,
        "train": _regression_metrics(x_train @ weights + y_mean, y_train, baseline),
        "test": _regression_metrics(x_test @ weights + y_mean, y_test, baseline),
    }


def catboost_kl_probe(
    features: list[torch.Tensor],
    targets: list[float],
    train_fraction: float,
    seed: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    task_type: str,
) -> dict:
    split = _feature_split(features, targets, train_fraction, seed)
    if not split["ok"]:
        return split

    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError:
        return {"ok": False, "reason": "catboost is not installed in this environment"}

    x = split["x"]
    x_train = split["x_train"].to(dtype=torch.float32).numpy()
    x_test = split["x_test"].to(dtype=torch.float32).numpy()
    y_train = split["y_train"].to(dtype=torch.float32).numpy()
    y_test = split["y_test"]
    baseline = float(split["y_train"].mean())

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=max(1, iterations),
        depth=max(1, depth),
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=seed,
        task_type=task_type,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)

    train_pred = torch.from_numpy(model.predict(x_train)).to(dtype=torch.float64)
    test_pred = torch.from_numpy(model.predict(x_test)).to(dtype=torch.float64)
    return {
        "ok": True,
        "pairs": int(len(split["y_train"]) + len(split["y_test"])),
        "hidden_dim": int(x.shape[1] // 2),
        "feature_dim": int(x.shape[1]),
        "train_n": int(len(split["y_train"])),
        "test_n": int(len(split["y_test"])),
        "iterations": int(max(1, iterations)),
        "depth": int(max(1, depth)),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2_leaf_reg),
        "task_type": task_type,
        "train": _regression_metrics(train_pred, split["y_train"], baseline),
        "test": _regression_metrics(test_pred, y_test, baseline),
    }


def print_linear_probe_summary(summary: dict) -> None:
    print("\n=== linear hidden-state KL predictor ===")
    if not summary["ok"]:
        print(f"skipped: {summary['reason']}")
        return

    print(
        f"features: [anchor_hidden, target_hidden] "
        f"hidden_dim={summary['hidden_dim']} feature_dim={summary['feature_dim']}"
    )
    print(
        f"split:    train={summary['train_n']} test={summary['test_n']} "
        f"ridge={summary['ridge']:.6g}"
    )
    _print_metrics(summary)


def print_catboost_probe_summary(summary: dict) -> None:
    print("\n=== CatBoost hidden-state KL predictor ===")
    if not summary["ok"]:
        print(f"skipped: {summary['reason']}")
        return

    print(
        f"features: [anchor_hidden, target_hidden] "
        f"hidden_dim={summary['hidden_dim']} feature_dim={summary['feature_dim']}"
    )
    print(
        f"split:    train={summary['train_n']} test={summary['test_n']} "
        f"iterations={summary['iterations']} depth={summary['depth']} "
        f"lr={summary['learning_rate']:.6g} l2={summary['l2_leaf_reg']:.6g} "
        f"task_type={summary['task_type']}"
    )
    _print_metrics(summary)


def _print_metrics(summary: dict) -> None:
    for split in ("train", "test"):
        metrics = summary[split]
        print(
            f"{split:>5}:   rmse={metrics['rmse']:.6f} "
            f"mae={metrics['mae']:.6f} r2={metrics['r2']:.4f} "
            f"baseline_rmse={metrics['baseline_rmse']:.6f}"
        )


def load_feature_file(path: Path) -> tuple[list[dict], list[torch.Tensor], list[float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = [row.detach().cpu() for row in payload["features"]]
    targets = [float(value) for value in payload["kl"].tolist()]
    return payload.get("records", []), features, targets

