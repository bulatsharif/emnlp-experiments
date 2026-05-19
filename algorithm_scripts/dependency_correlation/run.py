#!/usr/bin/env python
"""Run symmetric dependency/cosine correlation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from algorithm_scripts.dependency_correlation.collect import collect_sample
from algorithm_scripts.dependency_correlation.stats import print_table, summarize
from algorithm_scripts.dependency_correlation.tasks import (
    load_gsm8k,
    load_humaneval,
    load_shuffle,
)
from algorithm_scripts.kl_prediction.model import (
    load_dream_model,
    repo_root,
    resolve_shift_logits,
)


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
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(list(argv))


def load_samples(args):
    tasks = {task.strip().lower() for task in args.tasks.split(",") if task.strip()}
    samples = []
    if "shuffle" in tasks:
        samples.extend(
            load_shuffle(repo_root(), args.samples, args.sample_offset, args.sample_seed)
        )
    if "humaneval" in tasks:
        samples.extend(load_humaneval(args.humaneval_jsonl, args.samples, args.sample_offset))
    if "gsm8k" in tasks:
        samples.extend(load_gsm8k(args.gsm8k_jsonl, args.samples, args.sample_offset))
    return samples


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    samples = load_samples(args)
    model, tokenizer, mask_id, device = load_dream_model(args)
    args.shift_logits_resolved = resolve_shift_logits(args.shift_logits, args.model)

    records = []
    for sample in samples:
        print(f"{sample.task} sample {sample.metadata.get('sample_idx', 0)}")
        records.extend(collect_sample(model, tokenizer, sample, mask_id, device, args))

    rows = summarize(records)
    print_table(rows)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    if args.output_csv and rows:
        write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
