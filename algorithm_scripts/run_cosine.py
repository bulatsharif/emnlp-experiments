import torch.distributions as dists
from abc import ABC, abstractmethod
from .similarity import TokenSimilarity, AttentionTokenSimilarity, CosineTokenSimilarity
from .solver import GreedyUnmaskingSolver, DefaultUnmaskingSolver
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch
from .modeling_llada2_moe_pd import LLaDA2MoeModelLM
from .modeling_llada_pd import LLaDAModelLM
import os
import time
from collections import defaultdict
from .signal import calculate_signal, calculate_signal_gain

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def top_p_logits(logits, top_p=None):
    """
    Method that keeps only logits that are in the top p% of the distribution.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    """
    Method that keeps only logits that are in the top k of the distribution.
    """
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


@torch.no_grad()
def batch_generate_cosine(
    model,
    mask_token_id: int,
    eos_token_id: int,
    input_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    num_steps: int = 8,
    block_size: int = 32,
    temp: float = 0.75,
    top_p: float = 0.90,
    expand_token_id: int = None,
    top_k: int = None,
    similarity=None,
    solver=None,
    split_steps_across_blocks: bool = False,
    shift_logits: bool = True,
    alg_temp: float = 0.0,
    estimate_likelihood=False,
    statistics: Dict = None,
    **kwargs,
):
    """
    Generates with the provided model. It utilizes the hidden states of tokens to calculate cosine similarity between them.
    This allows to create penalization for dependent tokens, uplifting contextually independent tokens to generate in parallel.

    CAUTION: To make this method properly your work your model should return hidden states FROM LAST LAYER.

    Args:
    model: AutoModel - model to generate with.
    mask_token_id: int - token id of the mask. (for example - tokenizer.mask_token_id)
    input_ids: torch.Tensor - canvas of token ids in which we generate. Note that the method assumes you already incorporated here contiguous sequence of masks.
    attention_mask: torch.Tensor - attention mask for the model.
    num_steps: int - number of diffusion steps.
    block_size: int - size of the block in semi-autoregressive generation.
    temp: float - temperature for the logits.
    top_p: float - top p for the logits.
    expand_token_id: int - id of 'expand' token. For models such as Dream it is crucial to delete this token from generation.
    top_k: int - top k for the logits.
    cosine_on: bool - whether to utilize cosine penalization. If false, default unmasking behavior will be used.
    split_steps_across_blocks: bool - whether to split provided number of steps across blocks.
        For example, if num_steps = 16, block_size = 32, number of masks = 64, then we would have 2 blocks, and 8 steps per block.
    shift_logits: bool - whether to shift the logits. Used in model such as DreamOn because of its adapted nature. Should be False for LLaDa-like models.
    """
    B, _ = input_ids.shape
    assert B == 1, "Batch is not supported"
    response_mask_all = input_ids == mask_token_id
    tok_idx = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .unsqueeze(0)
        .repeat(input_ids.shape[0], 1)
    )

    if estimate_likelihood:
        neg_log_likelihood = 0
        marginal_neg_log_likelihood = 0
        number_of_calculated_tokens = 0

    # 1. Determine the canvas for generation
    start_indices = torch.argmax(response_mask_all.int(), dim=1)
    mask_lengths = torch.sum(response_mask_all, dim=1)
    gen_length = mask_lengths[0]

    run_statistics = defaultdict(int)
    # Задаем параметры provisional ремаскирования
    provision_on = kwargs.get(f"provision_on", False)
    if provision_on:
        provision_remasking_on = kwargs.get("provision_remasking_on", False)
        provisional_signal = kwargs.get("provisional_signal", None)
        signal_func = None
        if provisional_signal:
            if provisional_signal == "maxprob_margin":
                signal_func = calculate_signal
            elif provisional_signal == "gain_margin":
                signal_func = calculate_signal_gain

    # 2. Calculate number of blocks, break steps by blocks if needed
    num_blocks = mask_lengths[0] // block_size
    num_steps = num_steps // num_blocks if split_steps_across_blocks else num_steps
    tokens_per_step = block_size // num_steps
    assert (
        num_blocks * block_size == mask_lengths
    ), "Couldn't compose integer number of blocks from given masks and block size"
    trace = []

    # Создание маски для llada2.0
    if isinstance(model, (LLaDA2MoeModelLM)):
        # total_num_blocks - number of block including prompt devided into blocks
        prompt_length = input_ids.shape[-1] - mask_lengths[0]
        assert prompt_length % block_size == 0, "Need integer num of blocks"
        total_num_blocks = prompt_length // block_size + num_blocks
        block_mask = torch.tril(
            torch.ones(total_num_blocks, total_num_blocks, device=input_ids.device)
        )
        # Masks to masks attention
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_size, dim=0)
            .repeat_interleave(block_size, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()
        block_diffusion_attention_mask = torch.where(
            block_diffusion_attention_mask, 0.0, float("-inf")
        ).to(torch.bfloat16)

    for block_id in tqdm(
        range(num_blocks), desc="Generating Block", colour="green", position=0
    ):
        # 3. Determine the bounds and masks for current block
        block_start = start_indices + block_id * block_size
        block_end = start_indices + (block_id + 1) * block_size
        current_widow = (
            10**9
        )  # Для всех моделей, кроме некоторых делаем такую 'блочную' генерацию
        if isinstance(model, (LLaDA2MoeModelLM)) or True:
            current_widow = block_end
        x = input_ids[..., :current_widow].clone()
        tok_idx_ = tok_idx[..., :current_widow]
        current_block_mask = (tok_idx_ >= block_start.unsqueeze(1)) & (
            tok_idx_ < block_end.unsqueeze(1)
        )
        response_mask = current_block_mask & (x == mask_token_id)
        if isinstance(model, (LLaDA2MoeModelLM)):
            # block_end = block_end - block_end % block_size
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_widow, :current_widow
            ]
        else:
            cur_attn_mask = attention_mask
            if cur_attn_mask:
                cur_attn_mask = attention_mask[..., :current_widow]
        currently_unmasked = 0

        # Задаем параметры provisional ремаскирования на блок
        if provision_on:
            provisional_index = None
            provisional_margin_coefficient = kwargs.get(
                "provisional_margin_coefficient", None
            )
            unmask_at_last_step = kwargs.get("provisional_unmask_at_last_step", False)
            schedule_remasking_debts = kwargs.get(
                "provisional_schedule_remasking_debts", False
            )
            cur_step = 0

            if provision_remasking_on:
                assumed_number_of_steps = (response_mask.sum()) // tokens_per_step
                provision_schedule = [tokens_per_step] * assumed_number_of_steps
                remasking_debt = 0

        with tqdm(
            total=block_size,
            desc="Tokens Unmasked",
            colour="#00ffff",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            while True:
                # print(x)
                mask_index = current_block_mask & (x == mask_token_id)
                # print(f"{mask_index.sum()=}")
                # print(f"{cur_step=}")
                # print(f"{provision_schedule=}")
                # print(f"{provision_schedule[cur_step]=}")
                # if cur_step > 1:
                #     print(f"{provisional_tokens_chosen.shape=}")
                # print("----------------------")
                if mask_index.sum() == 0:
                    break

                number_transfer_tokens = tokens_per_step

                if provision_on:
                    if (
                        provision_remasking_on
                        and assumed_number_of_steps == 1
                        and unmask_at_last_step == True
                    ):
                        number_transfer_tokens += remasking_debt

                    if (
                        provision_remasking_on
                        and not unmask_at_last_step
                        and schedule_remasking_debts
                    ):
                        number_transfer_tokens = provision_schedule[cur_step]

                logits, masked_similarity_scores = similarity.infer_model(
                    input_ids=x,
                    position_ids=tok_idx_,
                    output_hidden_states=True,
                    output_attentions=(
                        False
                        if isinstance(model, (LLaDA2MoeModelLM, LLaDAModelLM))
                        else True  # Eager attention imp is strange in Llada models
                    ),
                    attention_mask=cur_attn_mask,
                    mask_token_id=mask_token_id,
                    current_block_mask=current_block_mask,
                    **kwargs,
                )

                if shift_logits:
                    response_mask_shifted = torch.cat(
                        [response_mask[:, 1:], response_mask[:, :1]], dim=1
                    )
                    logits[response_mask] = logits[response_mask_shifted].clone()

                mask_logits = logits[mask_index]
                # mask_logits[..., mask_token_id] -= 1e9
                if expand_token_id:
                    # Crucial to ban this token in case of DreamOn model.
                    mask_logits[..., expand_token_id] -= 1e9

                # 4. Sampling logits
                if temp > 0:
                    mask_logits = mask_logits / temp
                if (
                    top_p is not None
                    and top_p < 1
                    and not isinstance(model, (LLaDAModelLM,))
                ):
                    # На LladaModelLM top_p портит логиты
                    mask_logits = top_p_logits(mask_logits, top_p)
                if top_k is not None:
                    logits = top_k_logits(mask_logits, top_k)
                probs = torch.softmax(mask_logits, dim=-1)

                if temp > 0:
                    try:
                        x0 = dists.Categorical(probs=probs).sample()
                        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(
                            -1
                        )
                    except:
                        confidence, x0 = probs.max(dim=-1)
                else:
                    confidence, x0 = probs.max(dim=-1)

                confidence, number_transfer_tokens = solver.solve(
                    tokens=x0,
                    similarities=masked_similarity_scores,
                    confidence=confidence,
                    number_transfer_tokens=number_transfer_tokens,
                    input_ids=input_ids,
                    **kwargs,
                )

                currently_unmasked += number_transfer_tokens
                pbar.update(currently_unmasked)

                # 9. Transfer tokens to the canvas.
                full_confidence = torch.full_like(
                    x, -torch.inf, device=input_ids.device, dtype=confidence.dtype
                )
                full_confidence[mask_index] = confidence

                if alg_temp == 0.0:
                    _, transfer_index = torch.topk(
                        full_confidence, number_transfer_tokens
                    )
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(
                        full_confidence, num_samples=number_transfer_tokens
                    ).unsqueeze(0)

                x_ = (
                    torch.zeros_like(x, device=input_ids.device, dtype=torch.long)
                    + mask_token_id
                )
                x_[mask_index] = x0.clone()
                row_indices = (
                    torch.arange(input_ids.size(0), device=input_ids.device)
                    .unsqueeze(1)
                    .expand_as(transfer_index)
                )
                if mask_token_id in x_[row_indices, transfer_index].view(-1).tolist():
                    print(
                        f"Mask token were predicted on the {cur_step} step: {x_[row_indices, transfer_index]}"
                    )

                if estimate_likelihood:
                    assert B == 1
                    logits_ = logits.clone()
                    x_copy = x.clone()
                    for i, token in enumerate(transfer_index[0]):
                        logits_ = F.softmax(logits_, dim=-1)
                        prob_of_currently_demasked_token = logits_[
                            0, token, x_[0, token]
                        ]
                        # print(f"{logits_[row_indices, token].sort(descending=True)=}")
                        # print(f"{prob_of_currently_demasked_token=}")
                        if i == 0:
                            for j, token_marginal in enumerate(transfer_index[0]):
                                if x_[0, token] != eos_token_id:
                                    marginal_neg_log_likelihood += -torch.log(
                                        logits_[
                                            0, token_marginal, x_[0, token_marginal]
                                        ]
                                    ).cpu()

                        if x_[0, token] != eos_token_id:
                            number_of_calculated_tokens += 1
                            neg_log_likelihood += -torch.log(
                                prob_of_currently_demasked_token
                            ).cpu()

                        if i != len(transfer_index) - 1:
                            logits_, masked_similarity_scores = similarity.infer_model(
                                input_ids=x_copy,
                                position_ids=tok_idx_,
                                output_hidden_states=True,
                                output_attentions=(
                                    False
                                    if isinstance(
                                        model, (LLaDA2MoeModelLM, LLaDAModelLM)
                                    )
                                    else True  # Eager attention imp is strange in Llada models
                                ),
                                attention_mask=cur_attn_mask,
                                mask_token_id=mask_token_id,
                                current_block_mask=current_block_mask,
                                **kwargs,
                            )

                input_ids[row_indices, transfer_index] = x_[row_indices, transfer_index]
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

                if provision_on:
                    if provisional_index is not None:
                        provisional_logits_this_step = logits[
                            0, provisional_index.unsqueeze(0)
                        ].clone()[
                            0
                        ]  # [Y, V]
                        provisional_ids_this_step = input_ids[
                            0, provisional_index.unsqueeze(0)
                        ].clone()[0]
                        provisional_tokens_signal_mask, *signal_outputs = signal_func(
                            provisional_logits_this_step=provisional_logits_this_step,
                            provisional_logits_previous_step=provisional_logits_previous_step,
                            provisional_ids_this_step=provisional_ids_this_step,
                            **kwargs,  # includes provisional margin threshold and provisional gain threshold
                        )  # [Y]
                        provisional_tokens_chosen = provisional_index[
                            provisional_tokens_signal_mask
                        ]  # [Q]
                        if provisional_tokens_signal_mask.any() and (
                            provision_remasking_on == False
                            or assumed_number_of_steps != 1
                        ):
                            # [0, ...] since the method is not applicable for B > 1, therefore could hard-code like that.
                            if provision_remasking_on == True:
                                input_ids[0, provisional_tokens_chosen.unsqueeze(0)] = (
                                    mask_token_id
                                )
                                x[0, provisional_tokens_chosen.unsqueeze(0)] = (
                                    mask_token_id
                                )
                            else:
                                input_ids[0, provisional_tokens_chosen.unsqueeze(0)] = (
                                    signal_outputs[0]
                                )  # top1_id
                                x[0, provisional_tokens_chosen.unsqueeze(0)] = (
                                    signal_outputs[0]
                                )

                            if provision_remasking_on:
                                remasking_debt += provisional_tokens_chosen.numel()
                            run_statistics[
                                "revised_tokens"
                            ] += provisional_tokens_signal_mask.sum().item()

                    provisional_index = transfer_index[0].clone()  # [Y]
                    provisional_logits_previous_step = logits[
                        0, provisional_index
                    ].clone()  # [Y, V]

                    if provisional_margin_coefficient:
                        kwargs[
                            "provision_margin_threshold"
                        ] *= provisional_margin_coefficient

                    if provision_remasking_on:
                        assumed_number_of_steps -= 1
                        # if assumed_number_of_steps == 1 and unmask_at_last_step == True:
                        #     number_transfer_tokens -= remasking_debt
                        if schedule_remasking_debts and remasking_debt > 0:
                            loops = remasking_debt // assumed_number_of_steps
                            remainder = remasking_debt - loops * assumed_number_of_steps

                            for i in range(loops):
                                for j in range(
                                    (cur_step + 1),
                                    (cur_step + 1) + assumed_number_of_steps,
                                ):
                                    provision_schedule[j] += 1

                            for j in range((cur_step + 1), (cur_step + 1) + remainder):
                                provision_schedule[j] += 1

                            remasking_debt = 0

                run_statistics["number_of_steps"] += 1

                if provision_on:
                    cur_step += 1

                trace.append(input_ids.clone().detach().cpu().tolist())

    if statistics is not None:
        if estimate_likelihood:
            statistics["neg_log_likelihood"] = neg_log_likelihood
            statistics["avg_neg_log_likelihood"] = (
                neg_log_likelihood / number_of_calculated_tokens
            )
            statistics["marginal_neg_log_likelihood"] = marginal_neg_log_likelihood
            statistics["avg_marginal_neg_log_likelihood"] = (
                marginal_neg_log_likelihood / number_of_calculated_tokens
            )

    tqdm.write(f"Number of steps: {run_statistics["number_of_steps"]}")
    return input_ids, trace
