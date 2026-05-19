"""Collect hidden-state pair features and conditional KL targets."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from .model import forward_logits_hidden


def encode_prompt(tokenizer, sample) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            sample.messages,
            add_generation_prompt=True,
            tokenize=False,
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
    scored = []
    for i, j in itertools.combinations([int(x) for x in scheduled.tolist()], 2):
        scored.append((float((confidence[i] + confidence[j]).item()), i, j))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for _, i, j in scored[:max_pairs]]


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
    specs = [
        (*pair, *choose_anchor(*pair, confidence, anchor_mode))
        for pair in top_pairs(scheduled, confidence, max_pairs)
    ]

    records = []
    pair_features = []
    kl_targets = []
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        x_cond = x.repeat(len(batch), 1)
        for row, (_, _, anchor, _) in enumerate(batch):
            x_cond[row, mask_pos[anchor]] = int(top_ids[anchor])

        cond_logits, _ = forward_logits_hidden(model, x_cond, shift_logits)
        require_finite("conditioned logits", cond_logits)
        rows = torch.arange(len(batch), device=x.device)
        target_pos = torch.tensor(
            [int(mask_pos[target]) for _, _, _, target in batch],
            dtype=torch.long,
            device=x.device,
        )
        cond_log_p = log_probs_without_mask(cond_logits[rows, target_pos], mask_id)
        require_finite("conditioned log probabilities", cond_log_p)

        for row, (i, j, anchor, target) in enumerate(batch):
            kl = torch.sum(p[target] * (log_p[target] - cond_log_p[row])).item()
            if not math.isfinite(kl):
                raise RuntimeError("KL contains a non-finite value.")
            pair_features.append(torch.cat([hidden_features[anchor], hidden_features[target]]))
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
                    "anchor_token_id": int(top_ids[anchor]),
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

