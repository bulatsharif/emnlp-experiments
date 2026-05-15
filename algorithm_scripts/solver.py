from abc import ABC, abstractmethod
from scipy.sparse import csgraph
from numba import njit
import numpy as np
import torch


class UnmaskingSolver(ABC):
    @abstractmethod
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        **kwargs,
    ): ...


class DefaultUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        **kwargs,
    ):
        return confidence, number_transfer_tokens


class GreedyUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        function: str = "divide",
        **kwargs,
    ):
        B, _ = input_ids.shape

        batch_idx = torch.arange(
            B, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)
        chosen_tokens = torch.argmax(confidence.view(B, -1), dim=1).view(B, 1)

        # M = similarities.shape[-1]
        # j = torch.arange(M, device=similarities.device).view(1, 1, -1)  # [1, 1, M]
        # k = torch.arange(M, device=similarities.device).view(1, -1, 1)  # [1, M, 1]
        # distance = torch.abs(j - k) + 1  # [1, M, M]
        # scale = torch.pow(distance, 1 / 4)
        # scale = scale.expand(B, -1, -1)
        # similarities = similarities / scale

        for _ in range(number_transfer_tokens - 1):
            similarity_to_currently_selected_tokens = similarities[
                batch_idx, chosen_tokens
            ].sum(dim=1)
            if function == "substract":
                confidence_upd = (
                    confidence.view(B, -1) - similarity_to_currently_selected_tokens
                )
            elif function == "divide":
                confidence_upd = (
                    confidence.view(B, -1) / similarity_to_currently_selected_tokens
                )
            confidence_upd[batch_idx, chosen_tokens] = 1e-9
            chosen_tokens = torch.cat(
                [chosen_tokens, torch.argmax(confidence_upd, dim=1).view(B, 1)], dim=1
            )

        confidence_selected = torch.full(
            size=confidence.view(B, -1).shape, fill_value=1e-6, device=input_ids.device
        )
        confidence_selected[batch_idx, chosen_tokens] = 1e6
        confidence = confidence_selected.view(-1)

        return confidence, number_transfer_tokens


class KnapsackUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        aggregate_attention_strategy: str = "median",
        **kwargs,
    ):
        B, _ = input_ids.shape
        similarities = similarities[0]

        if B > 1:
            raise NotImplementedError(
                f"Knapsack unmasking is not implemented for Batched Input! Current Batch Size: {input_ids.shape[0]}.\n"
                + "Either use `batch_size`=1 or `default` unmasking strategy."
            )

        similarities *= 1000
        attn_received = similarities.sum(dim=-2).int()

        attn_received = attn_received.float().detach().cpu().numpy().astype(np.int32)
        confidence_np = confidence.float().detach().cpu().numpy()
        if confidence_np.dtype == np.float16:
            confidence_np = confidence_np.astype(np.float32)

        if aggregate_attention_strategy == "median":
            avg_attn_rcvd = np.ceil(np.median(attn_received))
        elif aggregate_attention_strategy == "mean":
            avg_attn_rcvd = np.ceil(np.mean(attn_received))

        confidence_kept, indices = self._solve_knapsack(
            attn_received, confidence_np, int(number_transfer_tokens * avg_attn_rcvd)
        )

        cur_number_transfer_tokens = len(indices)
        if cur_number_transfer_tokens < number_transfer_tokens:
            indices_new = []
            for idx in indices:
                indices_new.append(idx)

            _, sorted_idx = torch.sort(confidence, descending=True)
            for idx in sorted_idx:
                if idx not in indices_new:
                    indices_new.append(idx)
                    if len(indices_new) == number_transfer_tokens:
                        break

            indices = indices_new

        for idx in range(len(attn_received)):
            if idx not in indices:
                confidence[idx] = 1e-3
            else:
                confidence[idx] = 1e3

        return confidence, cur_number_transfer_tokens

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _solve_knapsack(weights, values, capacity):
        n = len(weights)
        dp = np.zeros(capacity + 1, dtype=values.dtype)
        keep = np.zeros((n + 1, capacity + 1), dtype=np.bool_)

        for i in range(1, n + 1):
            w_val = weights[i - 1]
            v_val = values[i - 1]
            for w in range(capacity, w_val - 1, -1):
                if dp[w - w_val] + v_val > dp[w]:
                    dp[w] = dp[w - w_val] + v_val
                    keep[i][w] = True

        selected_indices = []
        w = capacity
        for i in range(n, 0, -1):
            if keep[i][w]:
                selected_indices.append(i - 1)
                w -= w_val
                w = w - weights[i - 1] + weights[i - 1]

        res_indices = []
        curr_w = capacity
        for i in range(n, 0, -1):
            if keep[i][curr_w]:
                res_indices.append(i - 1)
                curr_w -= weights[i - 1]

        return dp[capacity], np.array(res_indices[::-1], dtype=np.int32)


class InverseKnapsackUnmaskingSolver(KnapsackUnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        aggregate_attention_strategy: str = "median",
        **kwargs,
    ):
        B, _ = input_ids.shape
        similarities = similarities[0]

        if B > 1:
            raise NotImplementedError(
                f"Knapsack unmasking is not implemented for Batched Input! Current Batch Size: {input_ids.shape[0]}.\n"
                + "Either use `batch_size`=1 or `default` unmasking strategy."
            )

        similarities *= 1000
        attn_received = similarities.sum(dim=-2).int()

        attn_received = (
            attn_received.float().detach().cpu().numpy().astype(np.int32) * -1
        )
        confidence_np = (
            confidence.float().detach().cpu().numpy().astype(np.int32) * 1000
        )
        confidence_np = np.max(confidence_np) - confidence_np
        if confidence_np.dtype == np.float16:
            confidence_np = confidence_np.astype(np.float32)

        if aggregate_attention_strategy == "median":
            avg_attn_rcvd = np.ceil(np.median(confidence_np))
        elif aggregate_attention_strategy == "mean":
            avg_attn_rcvd = np.ceil(np.mean(confidence_np))
        confidence_kept, indices = self._solve_knapsack(
            confidence_np, attn_received, int(number_transfer_tokens * avg_attn_rcvd)
        )

        cur_number_transfer_tokens = len(indices)
        if cur_number_transfer_tokens < number_transfer_tokens:
            indices_new = []
            for idx in indices:
                indices_new.append(idx)

            _, sorted_idx = torch.sort(confidence, descending=True)
            for idx in sorted_idx:
                if idx not in indices_new:
                    indices_new.append(idx)
                    if len(indices_new) == number_transfer_tokens:
                        break

            indices = indices_new

        for idx in range(len(attn_received)):
            if idx not in indices:
                confidence[idx] = 1e-3
            else:
                confidence[idx] = 1e3

        return confidence, cur_number_transfer_tokens


class ComponentsUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        neighboring_threshold: int = 0.05,
        **kwargs,
    ):

        similarities = torch.abs(similarities[0])

        if input_ids.shape[0] > 1:
            raise NotImplementedError(
                f"Components unmasking is not implemented for Batched Input! Current Batch Size: {x.shape[0]}.\n"
                + "Either use `batch_size`=1 or `default` unmasking strategy."
            )
        components = self._find_components(
            similarities, neighboring_threshold=neighboring_threshold
        )

        number_transfer_tokens = len(components)
        for component in components:
            component_confidences = confidence[component]
            best_idx = torch.argmax(component_confidences)
            confidence[component[best_idx]] = 1e3
            for i in range(len(component)):
                if i != best_idx:
                    confidence[component[i]] = 1e-3

        return confidence, number_transfer_tokens

    @staticmethod
    def _find_components(A_tensor, neighboring_threshold=25):
        adj_mask = A_tensor > neighboring_threshold

        adj_matrix = adj_mask.detach().cpu().numpy()

        n_components, labels = csgraph.connected_components(
            adj_matrix, directed=False, return_labels=True
        )

        sorted_indices = np.argsort(labels)
        sorted_labels = labels[sorted_indices]

        split_points = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
        components = np.split(sorted_indices, split_points)

        return [c.tolist() for c in components]


class DebugGreedyUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        **kwargs,
    ):
        B, _ = input_ids.shape

        batch_idx = torch.arange(
            B, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)
        chosen_tokens = torch.argmax(confidence.view(B, -1), dim=1).view(B, 1)

        for _ in range(number_transfer_tokens - 1):
            similarity_to_currently_selected_tokens = similarities[
                batch_idx, chosen_tokens
            ].sum(dim=1)
            confidence_upd = (
                confidence.view(B, -1) / similarity_to_currently_selected_tokens
            )
            confidence_upd[batch_idx, chosen_tokens] = 1e-9
            chosen_tokens = torch.cat(
                [chosen_tokens, torch.argmax(confidence_upd, dim=1).view(B, 1)], dim=1
            )

        confidence_selected = torch.full(
            size=confidence.view(B, -1).shape, fill_value=1e-6, device=input_ids.device
        )
        confidence_selected[batch_idx, chosen_tokens] = 1e6
        confidence = confidence_selected.view(-1)

        return confidence, number_transfer_tokens, chosen_tokens, similarities


class BeamGreedyUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        similarities: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        function: str = "divide",
        expand_k: int = 3,
        beam_size: int = 3,
        **kwargs,
    ):

        beam_size = min(confidence.shape[-1], beam_size)
        expand_k = min(confidence.shape[-1], expand_k)

        B, _ = input_ids.shape
        batch_idx = torch.arange(
            B, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)

        chosen_tokens = torch.argmax(confidence.view(B, -1), dim=1).view(B, 1)
        beam = [{"chosen": chosen_tokens, "score": confidence[chosen_tokens]}]

        # 7. Collect tokens which has least dependencies for already chosen tokens.
        for i in range(number_transfer_tokens - 1):
            new_beam = []
            for hyp in beam:
                chosen_at_beam = hyp["chosen"]
                scores_at_beam = hyp["score"]

                similarity_to_currently_selected_tokens = similarities[
                    batch_idx, chosen_at_beam
                ].sum(dim=1)
                if function == "substract":
                    confidence_upd = (
                        confidence.view(B, -1) - similarity_to_currently_selected_tokens
                    )
                elif function == "divide":
                    confidence_upd = (
                        confidence.view(B, -1) / similarity_to_currently_selected_tokens
                    )
                confidence_upd[batch_idx, chosen_at_beam] = 1e-9

                topk_scores, topk_ids = torch.topk(confidence_upd, k=expand_k, dim=1)
                for new_beam_idx in range(expand_k):
                    new_beam.append(
                        {
                            "chosen": torch.cat(
                                [
                                    chosen_at_beam,
                                    topk_ids[:, new_beam_idx].unsqueeze(1),
                                ],
                                dim=1,
                            ),
                            "score": scores_at_beam
                            * topk_scores[:, new_beam_idx].unsqueeze(1),
                        }
                    )

            for b in range(B):
                new_beam_for_this_sample = []
                for cur_beam in new_beam:
                    new_beam_for_this_sample.append(
                        {
                            "chosen": cur_beam["chosen"][b],
                            "score": cur_beam["score"][b].item(),
                        }
                    )
                new_beam_for_this_sample = sorted(
                    new_beam_for_this_sample, key=lambda x: x["score"], reverse=True
                )[:beam_size]
                for i_ in range(beam_size):
                    new_beam[i_]["chosen"][b] = new_beam_for_this_sample[i_]["chosen"]
                    new_beam[i_]["score"][b] = new_beam_for_this_sample[i_]["score"]

            beam = new_beam

        chosen_tokens = torch.full(
            size=(input_ids.shape[0], number_transfer_tokens),
            fill_value=0,
            device=input_ids.device,
        )

        for b in range(B):
            final_beam_for_this_sample = []
            for cur_beam in beam:
                final_beam_for_this_sample.append(
                    {
                        "chosen": cur_beam["chosen"][b],
                        "score": cur_beam["score"][b].item(),
                    }
                )
            final_beam_for_this_sample = max(
                final_beam_for_this_sample, key=lambda x: x["score"]
            )
            chosen_tokens[b] = final_beam_for_this_sample["chosen"]

        confidence_selected = torch.full(
            size=confidence.view(B, -1).shape, fill_value=1e-6, device=input_ids.device
        )
        confidence_selected[batch_idx, chosen_tokens] = 1e6
        confidence = confidence_selected.view(-1)

        return confidence, number_transfer_tokens


class DedublicationUnmaskingSolver(UnmaskingSolver):
    def solve(
        self,
        tokens: torch.Tensor = None,
        number_transfer_tokens: int = None,
        confidence: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        function: str = "divide",
        dedublication_alpha: float = 0.3,
        **kwargs,
    ):
        B, _ = input_ids.shape
        assert B == 1
        batch_idx = torch.arange(
            B, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)
        chosen_tokens = torch.argmax(confidence.view(B, -1), dim=1).view(B, 1)
        tokens = tokens.view(1, -1)
        sim_i = tokens.unsqueeze(-1)
        sim_j = tokens.unsqueeze(1)
        similarities = (sim_i == sim_j).long() * dedublication_alpha
        for _ in range(number_transfer_tokens - 1):
            similarity_to_currently_selected_tokens = (
                similarities[batch_idx, chosen_tokens].sum(dim=1) + 1
            )
            if function == "divide":
                confidence_upd = (
                    confidence.view(B, -1) / similarity_to_currently_selected_tokens
                )

            confidence_upd[batch_idx, chosen_tokens] = 1e-9
            chosen_tokens = torch.cat(
                [chosen_tokens, torch.argmax(confidence_upd, dim=1).view(B, 1)], dim=1
            )

        confidence_selected = torch.full(
            size=confidence.view(B, -1).shape, fill_value=1e-6, device=input_ids.device
        )
        confidence_selected[batch_idx, chosen_tokens] = 1e6
        confidence = confidence_selected.view(-1)

        return confidence, number_transfer_tokens
