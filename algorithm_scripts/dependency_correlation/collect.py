"""Collect cosine scores and symmetric KL dependency."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from algorithm_scripts.kl_prediction.collector import (
    encode_prompt,
    log_probs_without_mask,
    require_finite,
)
from algorithm_scripts.kl_prediction.model import forward_logits_hidden


def _cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(F.normalize(x.float(), p=2, dim=-1) @ F.normalize(x.float(), p=2, dim=-1).T)


def _attention_matrix(
    attentions,
    mask_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if attentions is None:
        size = int(mask_pos.numel())
        return torch.full((size, size), torch.nan, dtype=torch.float32, device=device)

    if isinstance(attentions, (tuple, list)):
        attentions = torch.stack([layer.float() for layer in attentions], dim=0)
    else:
        attentions = attentions.float()

    if attentions.dim() != 5:
        size = int(mask_pos.numel())
        return torch.full((size, size), torch.nan, dtype=torch.float32, device=device)

    mean_attention = attentions.mean(dim=0).mean(dim=1)[0]
    mask_attention = mean_attention.index_select(0, mask_pos).index_select(1, mask_pos)
    return 0.5 * (mask_attention + mask_attention.T)


def _top_pairs(
    scheduled: torch.Tensor,
    confidence: torch.Tensor,
    max_pairs: int,
    selection: str,
) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations([int(x) for x in scheduled.tolist()], 2))
    if selection == "any":
        return pairs[:max_pairs]

    scored = []
    for i, j in pairs:
        scored.append((float((confidence[i] + confidence[j]).item()), i, j))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for _, i, j in scored[:max_pairs]]


def collect_step(
    *,
    model,
    x: torch.Tensor,
    mask_pos: torch.Tensor,
    generation_start: int,
    task: str,
    sample_idx: int,
    step: int,
    mask_id: int,
    shift_logits: bool,
    tokens_per_step: int,
    max_pairs: int,
    pair_selection: str,
    batch_size: int,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    logits, hidden, attentions = forward_logits_hidden(
        model,
        x,
        shift_logits,
        return_attentions=True,
    )
    require_finite("base logits", logits)
    require_finite("base hidden states", hidden)

    masked_logits = logits[0, mask_pos].float()
    log_p = log_probs_without_mask(masked_logits, mask_id)
    p = log_p.exp()
    confidence, top_ids = p.max(dim=-1)
    confidence_order = torch.argsort(confidence, descending=True)
    if pair_selection == "any":
        scheduled = torch.arange(
            min(tokens_per_step, len(mask_pos)),
            device=mask_pos.device,
            dtype=torch.long,
        )
    else:
        scheduled = confidence_order[: min(tokens_per_step, len(mask_pos))]
    ranks = {int(idx): rank for rank, idx in enumerate(confidence_order.tolist())}

    hidden_cos = _cosine_matrix(hidden[0, mask_pos])
    logit_vec = masked_logits.clone()
    logit_vec[:, mask_id] = 0
    logit_cos = _cosine_matrix(logit_vec)
    attention_score = _attention_matrix(attentions, mask_pos, x.device)
    pairs = _top_pairs(scheduled, confidence, max_pairs, pair_selection)

    records = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        x_cond = x.repeat(2 * len(batch), 1)
        target_pos = []
        for row, (i, j) in enumerate(batch):
            x_cond[2 * row, mask_pos[j]] = int(top_ids[j])
            x_cond[2 * row + 1, mask_pos[i]] = int(top_ids[i])
            target_pos.extend([int(mask_pos[i]), int(mask_pos[j])])

        cond_logits, _, _ = forward_logits_hidden(model, x_cond, shift_logits)
        rows = torch.arange(len(target_pos), device=x.device)
        targets = torch.tensor(target_pos, dtype=torch.long, device=x.device)
        cond_log_p = log_probs_without_mask(cond_logits[rows, targets], mask_id)
        require_finite("conditioned log probabilities", cond_log_p)

        for row, (i, j) in enumerate(batch):
            kl_i = torch.sum(p[i] * (log_p[i] - cond_log_p[2 * row])).item()
            kl_j = torch.sum(p[j] * (log_p[j] - cond_log_p[2 * row + 1])).item()
            dependency = 0.5 * (kl_i + kl_j)
            if not math.isfinite(dependency):
                raise RuntimeError("symmetric KL contains a non-finite value")
            records.append(
                {
                    "task": task,
                    "sample_idx": sample_idx,
                    "step": step,
                    "i": int(mask_pos[i]) - generation_start,
                    "j": int(mask_pos[j]) - generation_start,
                    "rank_i": ranks[i],
                    "rank_j": ranks[j],
                    "token_i": int(top_ids[i]),
                    "token_j": int(top_ids[j]),
                    "confidence_i": float(confidence[i]),
                    "confidence_j": float(confidence[j]),
                    "pair_selection": pair_selection,
                    "attention_score": float(attention_score[i, j]),
                    "hidden_cosine": float(hidden_cos[i, j]),
                    "logit_cosine": float(logit_cos[i, j]),
                    "kl_i_given_j": float(kl_i),
                    "kl_j_given_i": float(kl_j),
                    "true_dependency": float(dependency),
                }
            )
    return records, scheduled, top_ids


@torch.no_grad()
def collect_sample(model, tokenizer, sample, mask_id: int, device: str, args):
    prompt_ids = encode_prompt(tokenizer, sample).to(device)
    masks = torch.full((1, args.max_new_tokens), mask_id, dtype=torch.long, device=device)
    x = torch.cat([prompt_ids, masks], dim=1)
    start = int((x[0] == mask_id).nonzero().flatten()[0])
    sample_idx = sample.metadata.get("sample_idx", 0)
    records = []

    max_steps = 1 if args.single_forward_only else args.generation_steps
    forward_mode = "single_forward" if args.single_forward_only else "iterative"

    for step in range(max_steps):
        mask_pos = (x[0] == mask_id).nonzero(as_tuple=False).flatten()
        if len(mask_pos) == 0:
            break
        rows, scheduled, top_ids = collect_step(
            model=model,
            x=x,
            mask_pos=mask_pos,
            generation_start=start,
            task=sample.task,
            sample_idx=sample_idx,
            step=step,
            mask_id=mask_id,
            shift_logits=args.shift_logits_resolved,
            tokens_per_step=args.tokens_per_step,
            max_pairs=args.max_pairs_per_step,
            pair_selection=args.pair_selection,
            batch_size=max(1, args.condition_batch_size),
        )
        for row in rows:
            row["forward_mode"] = forward_mode
        records.extend(rows)
        if args.single_forward_only:
            break
        x[0, mask_pos[scheduled]] = top_ids[scheduled]
    return records
