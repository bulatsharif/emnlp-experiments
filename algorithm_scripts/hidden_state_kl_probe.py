#!/usr/bin/env python
"""Hidden-state cosine vs. conditional KL during Shuffle generation.

The probe runs an actual confidence-based denoising trace: append 32 masks,
commit 4 highest-confidence tokens per step for 8 steps, and at each step test
only a few pairs among those scheduled tokens. For each tested pair, one token
is temporarily unmasked while the rest of the canvas is kept unchanged; the
reported KL is the distribution shift at the other still-masked token.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
    from .probe_summary import print_summary, summarize
except ImportError:
    from dream_probe_compat import (
        forward_logits_hidden,
        load_dream_model,
        resolve_shift_logits,
    )
    from parallelbench_shuffle import load_shuffle_sample
    from probe_summary import print_summary, summarize


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


def base_state(model, x: torch.Tensor, mask_pos: torch.Tensor, mask_id: int, shift: bool):
    logits, hidden = forward_logits_hidden(model, x, shift)
    log_probs = log_probs_without_mask(logits[0, mask_pos], mask_id)
    probs = log_probs.exp()
    confidence, top_ids = probs.max(dim=-1)
    hidden = F.normalize(hidden[0, mask_pos].float(), p=2, dim=-1)
    cosine = torch.abs(hidden @ hidden.T)
    return log_probs, probs, confidence, top_ids, cosine


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
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    log_p, p, confidence, top_ids, cosine = base_state(
        model, x, mask_pos, mask_id, shift_logits
    )
    order = torch.argsort(confidence, descending=True)
    scheduled = order[: min(tokens_per_step, len(mask_pos))]
    ranks = {int(idx): rank for rank, idx in enumerate(order.tolist())}

    specs = []
    for i, j in top_pairs(scheduled, confidence, max_pairs):
        anchor, target = choose_anchor(i, j, confidence, anchor_mode)
        specs.append((i, j, anchor, target, int(top_ids[anchor])))

    records = []
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        x_cond = x.repeat(len(batch), 1)
        for row, (_, _, anchor, _, token_id) in enumerate(batch):
            x_cond[row, mask_pos[anchor]] = token_id

        cond_logits, _ = forward_logits_hidden(model, x_cond, shift_logits)
        rows = torch.arange(len(batch), device=x.device)
        target_pos = torch.tensor(
            [int(mask_pos[target]) for _, _, _, target, _ in batch],
            dtype=torch.long,
            device=x.device,
        )
        cond_log_p = log_probs_without_mask(cond_logits[rows, target_pos], mask_id)

        for row, (i, j, anchor, target, token_id) in enumerate(batch):
            kl = torch.sum(p[target] * (log_p[target] - cond_log_p[row])).item()
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
    return records, scheduled, top_ids


@torch.no_grad()
def probe_sample(model, input_ids: torch.Tensor, mask_id: int, sample_idx: int, args):
    x = input_ids.clone()
    start = int((x[0] == mask_id).nonzero().flatten()[0])
    rows = []

    for step in range(args.generation_steps):
        mask_pos = (x[0] == mask_id).nonzero(as_tuple=False).flatten()
        if len(mask_pos) == 0:
            break

        records, scheduled, top_ids = pair_kl_records(
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
        x[0, mask_pos[scheduled]] = top_ids[scheduled]
    return rows


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
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--anchor", choices=["higher_confidence", "lower_confidence", "first"], default="higher_confidence")
    parser.add_argument("--shift-logits", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--sdpa-backend", default="auto", choices=["auto", "math"])
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[1]
    model, tokenizer, mask_id, device = load_dream_model(args)
    args.shift_logits_resolved = resolve_shift_logits(args.shift_logits, args.model)

    records = []
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
        records.extend(probe_sample(model, input_ids, mask_id, sample_idx, args))

    print_summary(summarize(records, args.bins))
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        print(f"\nwrote pair records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
