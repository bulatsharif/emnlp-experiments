#!/usr/bin/env python
"""Hidden-state pair features vs. conditional KL during Shuffle generation.

The probe runs an actual confidence-based denoising trace: append 32 masks,
commit 4 highest-confidence tokens per step for 8 steps, and at each step test
only a few pairs among those scheduled tokens. For each tested pair, one token
is temporarily unmasked while the rest of the canvas is kept unchanged; the
reported KL is the distribution shift at the other still-masked token. The
collected pair hidden states can then be used to fit a linear KL predictor.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

try:
    from .dream_probe_compat import (
        forward_logits_hidden,
        load_dream_model,
        resolve_shift_logits,
    )
    from .parallelbench_shuffle import load_shuffle_sample
except ImportError:
    from dream_probe_compat import (
        forward_logits_hidden,
        load_dream_model,
        resolve_shift_logits,
    )
    from parallelbench_shuffle import load_shuffle_sample


DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[1] / "Dream-org/Dream-Coder-v0-Instruct-7B"
)


def encode_prompt(tokenizer, sample) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            sample.messages, add_generation_prompt=True, tokenize=False
        )
    else:
        text = sample.messages[-1]["content"]
    return tokenizer(text, return_tensors="pt").input_ids


def encode_len(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(ids[0] if ids and isinstance(ids[0], list) else ids)


def log_probs_without_mask(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    logits = logits.float().clone()
    logits[..., mask_id] = torch.finfo(logits.dtype).min
    return F.log_softmax(logits, dim=-1)


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise RuntimeError(
        f"{name} contains non-finite values (nan={nan_count}, inf={inf_count})."
    )


def top_pairs(
    scheduled: torch.Tensor,
    confidence: torch.Tensor,
    max_pairs: int,
) -> list[tuple[int, int]]:
    pairs = []
    for i, j in itertools.combinations([int(x) for x in scheduled.tolist()], 2):
        pairs.append((float((confidence[i] + confidence[j]).item()), i, j))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for _, i, j in pairs[:max_pairs]]


def choose_anchor(i: int, j: int, confidence: torch.Tensor, mode: str) -> tuple[int, int]:
    if mode == "first":
        return i, j
    if mode == "lower_confidence":
        return (i, j) if confidence[i] <= confidence[j] else (j, i)
    return (i, j) if confidence[i] >= confidence[j] else (j, i)


def pair_kl_records(
    *,
    model,
    x: torch.Tensor,
    mask_pos: torch.Tensor,
    generation_start: int,
    step: int,
    sample_idx: int,
    mask_id: int,
    shift_logits: bool,
    anchor_mode: str,
    batch_size: int,
    max_pairs: int,
    tokens_per_step: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, list[torch.Tensor], list[float]]:
    logits, hidden = forward_logits_hidden(model, x, shift_logits)
    require_finite("base logits", logits)
    require_finite("base hidden states", hidden)
    log_p = log_probs_without_mask(logits[0, mask_pos], mask_id)
    require_finite("base log probabilities", log_p)
    p = log_p.exp()
    confidence, top_ids = p.max(dim=-1)
    hidden = hidden[0, mask_pos].float()
    hidden_features = hidden.detach().cpu()
    hidden_norm = F.normalize(hidden, p=2, dim=-1)
    cosine = torch.abs(hidden_norm @ hidden_norm.T)
    require_finite("base confidences", confidence)
    require_finite("base cosine matrix", cosine)

    order = torch.argsort(confidence, descending=True)
    scheduled = order[: min(tokens_per_step, len(mask_pos))]
    ranks = {int(idx): rank for rank, idx in enumerate(order.tolist())}

    specs = []
    for i, j in top_pairs(scheduled, confidence, max_pairs):
        anchor, target = choose_anchor(i, j, confidence, anchor_mode)
        specs.append((i, j, anchor, target, int(top_ids[anchor])))

    records = []
    pair_features = []
    kl_targets = []
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        x_cond = x.repeat(len(batch), 1)
        for row, (_, _, anchor, _, token_id) in enumerate(batch):
            x_cond[row, mask_pos[anchor]] = token_id

        cond_logits, _ = forward_logits_hidden(model, x_cond, shift_logits)
        require_finite("conditioned logits", cond_logits)
        rows = torch.arange(len(batch), device=x.device)
        target_pos = torch.tensor(
            [int(mask_pos[target]) for _, _, _, target, _ in batch],
            dtype=torch.long,
            device=x.device,
        )
        cond_log_p = log_probs_without_mask(cond_logits[rows, target_pos], mask_id)
        require_finite("conditioned log probabilities", cond_log_p)

        for row, (i, j, anchor, target, token_id) in enumerate(batch):
            kl = torch.sum(p[target] * (log_p[target] - cond_log_p[row])).item()
            if not math.isfinite(kl):
                raise RuntimeError("KL contains a non-finite value.")
            pair_features.append(
                torch.cat([hidden_features[anchor], hidden_features[target]])
            )
            kl_targets.append(float(kl))
            records.append(
                {
                    "task": "waiting_line/shuffle",
                    "sample_idx": sample_idx,
                    "step": step,
                    "pair_i": int(mask_pos[i]) - generation_start,
                    "pair_j": int(mask_pos[j]) - generation_start,
                    "rank_i": ranks[i],
                    "rank_j": ranks[j],
                    "anchor_idx": int(mask_pos[anchor]) - generation_start,
                    "target_idx": int(mask_pos[target]) - generation_start,
                    "anchor_pos": int(mask_pos[anchor]),
                    "target_pos": int(mask_pos[target]),
                    "anchor_token_id": token_id,
                    "anchor_confidence": float(confidence[anchor]),
                    "target_confidence": float(confidence[target]),
                    "cosine": float(cosine[i, j]),
                    "kl": float(kl),
                }
            )
    return records, scheduled, top_ids, pair_features, kl_targets


@torch.no_grad()
def probe_sample(model, input_ids: torch.Tensor, mask_id: int, sample_idx: int, args):
    x = input_ids.clone()
    start = int((x[0] == mask_id).nonzero().flatten()[0])
    rows = []
    features = []
    targets = []

    for step in range(args.generation_steps):
        mask_pos = (x[0] == mask_id).nonzero(as_tuple=False).flatten()
        if len(mask_pos) == 0:
            break

        records, scheduled, top_ids, pair_features, kl_targets = pair_kl_records(
            model=model,
            x=x,
            mask_pos=mask_pos,
            generation_start=start,
            step=step,
            sample_idx=sample_idx,
            mask_id=mask_id,
            shift_logits=args.shift_logits_resolved,
            anchor_mode=args.anchor,
            batch_size=max(1, args.condition_batch_size),
            max_pairs=args.max_pairs_per_step,
            tokens_per_step=args.tokens_per_step,
        )
        rows.extend(records)
        features.extend(pair_features)
        targets.extend(kl_targets)
        x[0, mask_pos[scheduled]] = top_ids[scheduled]
    return rows, features, targets


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
    centered = target - target.mean()
    sst = torch.sum(centered.square())
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
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return {
        "ok": True,
        "x": x,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
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
    y_centered = y_train - y_mean
    ridge = max(float(ridge), 0.0)

    gram = x_train @ x_train.T
    if ridge > 0.0:
        gram = gram + ridge * torch.eye(len(x_train), dtype=gram.dtype)

    try:
        alpha = torch.linalg.solve(gram, y_centered)
    except RuntimeError:
        alpha = torch.linalg.lstsq(gram, y_centered[:, None]).solution[:, 0]

    weights = x_train.T @ alpha
    train_pred = x_train @ weights + y_mean
    test_pred = x_test @ weights + y_mean
    baseline = float(y_mean)

    return {
        "ok": True,
        "pairs": int(len(y_train) + len(y_test)),
        "hidden_dim": int(x.shape[1] // 2),
        "feature_dim": int(x.shape[1]),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "ridge": ridge,
        "train": _regression_metrics(train_pred, y_train, baseline),
        "test": _regression_metrics(test_pred, y_test, baseline),
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
    y_train_t = split["y_train"]
    baseline = float(y_train_t.mean())

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
        "train": _regression_metrics(train_pred, y_train_t, baseline),
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
    for split in ("train", "test"):
        metrics = summary[split]
        print(
            f"{split:>5}:   rmse={metrics['rmse']:.6f} "
            f"mae={metrics['mae']:.6f} r2={metrics['r2']:.4f} "
            f"baseline_rmse={metrics['baseline_rmse']:.6f}"
        )


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
    for split in ("train", "test"):
        metrics = summary[split]
        print(
            f"{split:>5}:   rmse={metrics['rmse']:.6f} "
            f"mae={metrics['mae']:.6f} r2={metrics['r2']:.4f} "
            f"baseline_rmse={metrics['baseline_rmse']:.6f}"
        )


def load_feature_file(path: Path) -> tuple[list[dict], list[torch.Tensor], list[float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_tensor = payload["features"]
    targets = [float(value) for value in payload["kl"].tolist()]
    records = payload.get("records", [])
    features = [row.detach().cpu() for row in feature_tensor]
    return records, features, targets


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-steps", type=int, default=8)
    parser.add_argument("--tokens-per-step", type=int, default=4)
    parser.add_argument("--max-pairs-per-step", type=int, default=4)
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument(
        "--anchor",
        choices=["higher_confidence", "lower_confidence", "first"],
        default="higher_confidence",
    )
    parser.add_argument("--linear-train-fraction", type=float, default=0.8)
    parser.add_argument("--linear-ridge", type=float, default=1.0)
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--skip-linear-probe", action="store_true")
    parser.add_argument("--train-catboost", action="store_true")
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--shift-logits", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--sdpa-backend", default="auto", choices=["auto", "math"])
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-features", type=Path)
    parser.add_argument("--input-features", type=Path)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[1]

    records = []
    features = []
    targets = []

    if args.input_features:
        records, features, targets = load_feature_file(args.input_features)
        print(f"loaded pair features from {args.input_features}")
    else:
        model, tokenizer, mask_id, device = load_dream_model(args)
        args.shift_logits_resolved = resolve_shift_logits(args.shift_logits, args.model)

        for sample_idx in range(args.sample_offset, args.sample_offset + args.samples):
            sample = load_shuffle_sample(repo_root, sample_idx, args.sample_seed)
            prompt_ids = encode_prompt(tokenizer, sample).to(device)
            masks = torch.full(
                (1, args.max_new_tokens), mask_id, dtype=torch.long, device=device
            )
            input_ids = torch.cat([prompt_ids, masks], dim=1)
            print(
                f"shuffle sample {sample_idx}: target_tokens={encode_len(tokenizer, sample.target)} "
                f"generated_masks={args.max_new_tokens} metadata={sample.metadata}"
            )
            sample_records, sample_features, sample_targets = probe_sample(
                model, input_ids, mask_id, sample_idx, args
            )
            records.extend(sample_records)
            features.extend(sample_features)
            targets.extend(sample_targets)

    print_collection_summary(records)
    if not args.skip_linear_probe:
        print_linear_probe_summary(
            linear_kl_probe(
                features,
                targets,
                args.linear_train_fraction,
                args.linear_ridge,
                args.linear_seed,
            )
        )
    if args.train_catboost:
        print_catboost_probe_summary(
            catboost_kl_probe(
                features,
                targets,
                args.linear_train_fraction,
                args.linear_seed,
                args.catboost_iterations,
                args.catboost_depth,
                args.catboost_learning_rate,
                args.catboost_l2_leaf_reg,
                args.catboost_task_type,
            )
        )
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        print(f"\nwrote pair records to {args.output_jsonl}")
    if args.output_features:
        args.output_features.parent.mkdir(parents=True, exist_ok=True)
        feature_tensor = torch.stack(features) if features else torch.empty(0, 0)
        torch.save(
            {
                "features": feature_tensor,
                "kl": torch.tensor(targets, dtype=torch.float32),
                "records": records,
                "feature_order": "anchor_hidden_then_target_hidden",
            },
            args.output_features,
        )
        print(f"wrote pair features to {args.output_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
