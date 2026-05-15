from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch
import time


class TokenSimilarity(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _get_similarities(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ): ...

    def infer_model(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError(
                f"Model {type(self.model).__name__} does not return logits!"
            )

        similarities = self._get_similarities(
            outputs, input_ids, mask_token_id, current_block_mask, **kwargs
        )
        return logits, similarities


class DefaultInfer(TokenSimilarity):
    def _get_similarities(*args, **kwargs):
        return None


class CosineTokenSimilarity(TokenSimilarity):
    def _get_similarities(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, *_ = input_ids.shape

        # 1. Retrieve hidden states
        last_hidden_states = getattr(outputs, "hidden_states", None)
        if last_hidden_states is None:
            raise ValueError(
                f"[ERROR] Model {type(self.model).__name__} does not return hidden states!"
            )
        if isinstance(last_hidden_states, tuple):
            print(
                f"[CRITICAL WARNING] Model {type(self.model).__name__} returns tuple for hidden states!\n\
                             Most likely, you forget to make the model return hidden states only from LAST layer.\n\
                             Automatically choosing last layer, which could lead to undefined behavior. "
            )
            last_hidden_states = last_hidden_states[-1]

        # 2. Calculate similarities
        last_hidden_states_normalized = F.normalize(last_hidden_states, p=2, dim=-1)
        similarity_scores = torch.bmm(
            last_hidden_states_normalized,
            last_hidden_states_normalized.transpose(-2, -1),
        )

        # 3. Focus only on mask tokens.
        if current_block_mask is None:
            print(f"[WARNING] Current Block Mask weren't provided. ")
            current_block_mask = torch.ones_like(
                input_ids, dtype=torch.bool, device=input_ids.device
            )
        rows_to_keep = (input_ids == mask_token_id) & current_block_mask
        current_number_of_masks = rows_to_keep.sum(dim=1)[0].item()
        intersection_mask = rows_to_keep.unsqueeze(2) & rows_to_keep.unsqueeze(1)
        masked_similarity_scores = torch.abs(
            similarity_scores[intersection_mask].view(
                B, current_number_of_masks, current_number_of_masks
            )
        )

        return masked_similarity_scores


class AttentionTokenSimilarity(TokenSimilarity):
    def _get_similarities(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        head_pooling_strategy: str = "mean",
        **kwargs,
    ):
        B, *_ = input_ids.shape

        # 1. Get attentions from the model outptus
        attn_weights = getattr(outputs, "attentions", None)
        if attn_weights is None:
            raise ValueError(
                f"[ERROR] Model {type(self.model).__name__} does not return attentions!"
            )
        if isinstance(attn_weights, tuple):
            attn_weights = torch.stack(attn_weights, dim=0)

        # 2. Aggregate different attention layers
        attn_weights = self._aggregate_attention_across_layers(attn_weights, **kwargs)

        # 3. Pool different attention heads
        _, _, num_queries, num_keys = attn_weights.shape
        masked_attention_matrix = self._pool_attention_heads(
            attn_weights, head_pooling_strategy
        )

        # 4. Focus only on mask tokens.
        rows_to_keep = (input_ids == mask_token_id) & current_block_mask
        current_number_of_masks = rows_to_keep.sum(dim=1)[0].item()
        intersection_mask = rows_to_keep.unsqueeze(2) & rows_to_keep.unsqueeze(1)
        masked_attention_matrix = masked_attention_matrix[intersection_mask].view(
            B, current_number_of_masks, current_number_of_masks
        )

        return masked_attention_matrix

    def _aggregate_attention_across_layers(
        self,
        attn_weights: torch.Tensor,
        layers_aggregating_strategy_start: int = None,
        layers_aggregating_strategy_end: int = None,
        **kwargs,
    ):
        if (
            layers_aggregating_strategy_start is not None
            and layers_aggregating_strategy_end is None
        ):
            attn_weights = attn_weights[layers_aggregating_strategy_start:, ...]
        elif (
            layers_aggregating_strategy_start is None
            and layers_aggregating_strategy_end is not None
        ):
            attn_weights = attn_weights[:layers_aggregating_strategy_end, ...]
        elif (
            layers_aggregating_strategy_end is not None
            and layers_aggregating_strategy_start is not None
        ):
            attn_weights = attn_weights[
                layers_aggregating_strategy_start:layers_aggregating_strategy_end, ...
            ]

        attn_weights = torch.mean(attn_weights, dim=0)
        return attn_weights

    def _pool_attention_heads(
        self, attention: torch.Tensor, head_pooling_strategy: str
    ):
        """
        Pools attention from heads into one matrix.

        Args:
            attention: (torch.Tensor) tensor which contains the attention values for each head
            head_pooling_strategy: (str) strategy to pool the heads. Can be 'mean', 'max' or 'concat'
        Returns:
            attention: (torch.Tensor) attention pulled from different heads into one attention matrix.
        """

        if head_pooling_strategy == "mean":
            return attention.mean(dim=1)
        elif head_pooling_strategy == "max":
            return attention.max(dim=1)[0]
        elif head_pooling_strategy == "sum":
            return attention.sum(dim=1)
        else:
            raise ValueError(
                f"Unsupported head pooling strategy. Available strategies: {["mean", "max", "sum"]}"
            )


class InterpolateAttentionCosineSimilarity(TokenSimilarity):
    def _get_similarities(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        head_pooling_strategy: str = "mean",
        interpolate_strategy: str = "alpha",
        interpolate_alpha: float = 0.5,
        interpolate_beta: float = 0.5,
        **kwargs,
    ):
        attn_matrix = self._get_similarities_attention(
            outputs=outputs,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            current_block_mask=current_block_mask,
            head_pooling_strategy=head_pooling_strategy,
            **kwargs,
        )

        cosine_matrix = self._get_similarities_cosine(
            outputs=outputs,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            current_block_mask=current_block_mask,
            head_pooling_strategy=head_pooling_strategy,
            **kwargs,
        )

        if interpolate_strategy == "alpha":
            return interpolate_alpha * attn_matrix + interpolate_beta * cosine_matrix
        else:
            return torch.max(attn_matrix, cosine_matrix)

    def _get_similarities_cosine(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        head_pooling_strategy: str = "mean",
        **kwargs,
    ):
        B, *_ = input_ids.shape

        # 1. Retrieve hidden states
        last_hidden_states = getattr(outputs, "hidden_states", None)
        if last_hidden_states is None:
            raise ValueError(
                f"[ERROR] Model {type(self.model).__name__} does not return hidden states!"
            )
        if isinstance(last_hidden_states, tuple):
            print(
                f"[CRITICAL WARNING] Model {type(self.model).__name__} returns tuple for hidden states!\n\
                             Most likely, you forget to make the model return hidden states only from LAST layer.\n\
                             Automatically choosing last layer, which could lead to undefined behavior. "
            )
            last_hidden_states = last_hidden_states[-1]

        # 2. Calculate similarities
        last_hidden_states_normalized = F.normalize(last_hidden_states, p=2, dim=-1)
        similarity_scores = torch.bmm(
            last_hidden_states_normalized,
            last_hidden_states_normalized.transpose(-2, -1),
        )

        # 3. Focus only on mask tokens.
        if current_block_mask is None:
            print(f"[WARNING] Current Block Mask weren't provided. ")
            current_block_mask = torch.ones_like(
                input_ids, dtype=torch.bool, device=input_ids.device
            )
        rows_to_keep = (input_ids == mask_token_id) & current_block_mask
        current_number_of_masks = rows_to_keep.sum(dim=1)[0].item()
        intersection_mask = rows_to_keep.unsqueeze(2) & rows_to_keep.unsqueeze(1)
        masked_similarity_scores = torch.abs(
            similarity_scores[intersection_mask].view(
                B, current_number_of_masks, current_number_of_masks
            )
        )

        return masked_similarity_scores

    def _get_similarities_attention(
        self,
        outputs: Union[Tuple, torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        mask_token_id: int = None,
        current_block_mask: Optional[torch.Tensor] = None,
        head_pooling_strategy: str = "mean",
        **kwargs,
    ):
        B, *_ = input_ids.shape

        # 1. Get attentions from the model outptus
        attn_weights = getattr(outputs, "attentions", None)
        if attn_weights is None:
            raise ValueError(
                f"[ERROR] Model {type(self.model).__name__} does not return attentions!"
            )
        if isinstance(attn_weights, tuple):
            attn_weights = torch.stack(attn_weights, dim=0)

        # 2. Aggregate different attention layers
        attn_weights = self._aggregate_attention_across_layers(attn_weights, **kwargs)

        # 3. Pool different attention heads
        _, _, num_queries, num_keys = attn_weights.shape
        masked_attention_matrix = self._pool_attention_heads(
            attn_weights, head_pooling_strategy
        )

        # 4. Focus only on mask tokens.
        rows_to_keep = (input_ids == mask_token_id) & current_block_mask
        current_number_of_masks = rows_to_keep.sum(dim=1)[0].item()
        intersection_mask = rows_to_keep.unsqueeze(2) & rows_to_keep.unsqueeze(1)
        masked_attention_matrix = masked_attention_matrix[intersection_mask].view(
            B, current_number_of_masks, current_number_of_masks
        )

        return masked_attention_matrix

    def _aggregate_attention_across_layers(
        self,
        attn_weights: torch.Tensor,
        layers_aggregating_strategy_start: int = None,
        layers_aggregating_strategy_end: int = None,
        layers_aggregating_ids: List[int] = None,
        **kwargs,
    ):
        if layers_aggregating_ids is not None:
            layers_aggregating_ids_tensor = torch.tensor(
                layers_aggregating_ids, dtype=torch.long, device=attn_weights.device
            )
            attn_weights = attn_weights[layers_aggregating_ids_tensor, ...]
        else:
            if (
                layers_aggregating_strategy_start is not None
                and layers_aggregating_strategy_end is None
            ):
                attn_weights = attn_weights[layers_aggregating_strategy_start:, ...]
            elif (
                layers_aggregating_strategy_start is None
                and layers_aggregating_strategy_end is not None
            ):
                attn_weights = attn_weights[:layers_aggregating_strategy_end, ...]
            elif (
                layers_aggregating_strategy_end is not None
                and layers_aggregating_strategy_start is not None
            ):
                attn_weights = attn_weights[
                    layers_aggregating_strategy_start:layers_aggregating_strategy_end,
                    ...,
                ]

        attn_weights = torch.mean(attn_weights, dim=0)
        return attn_weights

    def _pool_attention_heads(
        self, attention: torch.Tensor, head_pooling_strategy: str
    ):
        """
        Pools attention from heads into one matrix.

        Args:
            attention: (torch.Tensor) tensor which contains the attention values for each head
            head_pooling_strategy: (str) strategy to pool the heads. Can be 'mean', 'max' or 'concat'
        Returns:
            attention: (torch.Tensor) attention pulled from different heads into one attention matrix.
        """

        if head_pooling_strategy == "mean":
            return attention.mean(dim=1)
        elif head_pooling_strategy == "max":
            return attention.max(dim=1)[0]
        elif head_pooling_strategy == "sum":
            return attention.sum(dim=1)
        else:
            raise ValueError(
                f"Unsupported head pooling strategy. Available strategies: {["mean", "max", "sum"]}"
            )
