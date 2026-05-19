#!/usr/bin/env python
"""Collect pairwise KL targets and fit hidden-state KL predictors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kl_prediction.collector import encode_len, encode_prompt, probe_sample
    from kl_prediction.model import load_dream_model, repo_root, resolve_shift_logits
    from kl_prediction.regression import (
        catboost_kl_probe,
        linear_kl_probe_by_task,
        load_feature_file,
        print_catboost_probe_summary,
        print_collection_summary,
        print_linear_probe_by_task_summary,
    )
    from kl_prediction.tasks import load_samples
else:
    from .collector import encode_len, encode_prompt, probe_sample
    from .model import load_dream_model, repo_root, resolve_shift_logits
    from .regression import (
        catboost_kl_probe,
        linear_kl_probe_by_task,
        load_feature_file,
        print_catboost_probe_summary,
        print_collection_summary,
        print_linear_probe_by_task_summary,
    )
    from .tasks import load_samples


DEFAULT_MODEL = str(repo_root() / "Dream-org/Dream-Coder-v0-Instruct-7B")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", default="shuffle")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--humaneval-jsonl", type=Path)
    parser.add_argument("--gsm8k-jsonl", type=Path)
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
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--sdpa-backend", default="auto", choices=["auto", "math"])
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-features", type=Path)
    parser.add_argument("--input-features", type=Path)
    return parser.parse_args(list(argv))


def collect_features(args) -> tuple[list[dict], list[torch.Tensor], list[float]]:
    model, tokenizer, mask_id, device = load_dream_model(args)
    args.shift_logits_resolved = resolve_shift_logits(args.shift_logits, args.model)

    records = []
    features = []
    targets = []
    for sample in load_samples(args, repo_root()):
        sample_idx = sample.metadata.get("sample_idx", 0)
        prompt_ids = encode_prompt(tokenizer, sample).to(device)
        masks = torch.full((1, args.max_new_tokens), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([prompt_ids, masks], dim=1)
        print(
            f"{sample.task} sample {sample_idx}: "
            f"target_tokens={encode_len(tokenizer, sample.target)} "
            f"generated_masks={args.max_new_tokens} metadata={sample.metadata}"
        )
        rows, pair_features, kl_targets = probe_sample(
            model,
            input_ids,
            mask_id,
            sample_idx,
            sample.task,
            args,
        )
        records.extend(rows)
        features.extend(pair_features)
        targets.extend(kl_targets)
    return records, features, targets


def write_outputs(
    args,
    records: list[dict],
    features: list[torch.Tensor],
    targets: list[float],
) -> None:
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.input_features:
        records, features, targets = load_feature_file(args.input_features)
        print(f"loaded pair features from {args.input_features}")
    else:
        records, features, targets = collect_features(args)

    print_collection_summary(records)
    if not args.skip_linear_probe:
        print_linear_probe_by_task_summary(
            linear_kl_probe_by_task(
                records,
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
    write_outputs(args, records, features, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
