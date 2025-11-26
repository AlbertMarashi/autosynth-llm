"""
Fused Sparse Cross-Entropy Integration Module (Memory Optimized)

NO redundant W_T storage - uses weight.T directly (cuBLAS handles transpose).

Memory usage:
- Weights: [V, D] stored ONCE (or zero extra if tied with embeddings)
- No [D, V] transpose copy
- Logits: Never materialized (chunked computation)

Author: Lord Albert Marashi
"""
import torch
import torch.nn as nn
from typing import Optional, List
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from autosynth.modules.fused_kernel import chunked_ce_forward, chunked_ce_backward

# ===================
# LM HEAD FUNCTION
# ===================

class FusedSparseCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,                            # context object
        h: torch.Tensor,                # [N, D] - hidden states
        W: torch.Tensor,                # [V, D] - NOT transposed
        trainable_buffer: torch.Tensor, # [V_t, D] - trainable buffer
        labels: torch.Tensor,           # [N] - target indices
        loss_weights: torch.Tensor,     # [N] - per-token weights
        trainable_start_idx: int,       # index of the first trainable token
        V_CHUNK: int,                   # chunk size for the weight matrix
    ) -> torch.Tensor:

        loss, lse, weight_sum = chunked_ce_forward(
            h, W, labels, loss_weights, V_CHUNK
        )

        weight_sum_t = weight_sum.clone() if isinstance(weight_sum, torch.Tensor) else torch.tensor(weight_sum, device=h.device)

        ctx.save_for_backward(h, W, labels, loss_weights, lse, weight_sum_t)
        ctx.trainable_start_idx = trainable_start_idx
        ctx.V_CHUNK = V_CHUNK

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with FP16 accumulation.
        """
        h, W, labels, loss_weights, lse, weight_sum_t = ctx.saved_tensors

        grad_h, grad_W_trainable = chunked_ce_backward(
            h, W, labels, loss_weights, lse, weight_sum_t,
            ctx.trainable_start_idx, ctx.V_CHUNK
        )

        grad_h = grad_h * grad_output
        grad_W_trainable = grad_W_trainable * grad_output

        return grad_h, None, grad_W_trainable, None, None, None, None, None


# ======================
# SPARSE TIED WEIGHTS
# ======================

class SparseTiedWeights(nn.Module):
    """
    Weight matrix with trainable tail portion.
    Compatible with existing SparseEmbedding code.
    """

    def __init__(self, weight: torch.Tensor, trainable_start_idx: int):
        super().__init__()
        self.trainable_start_idx = trainable_start_idx
        self.weight = weight  # Original reference

        # Trainable buffer (same interface as your existing code)
        self.trainable_buffer = nn.Parameter(
            self.weight[trainable_start_idx:].clone().float()
        )

    def sync(self):
        with torch.no_grad():
            self.weight[self.trainable_start_idx:] = self.trainable_buffer.to(self.weight.dtype)




class FusedSparseLMHead(nn.Module):
    """
    Fused language model head - computes loss directly without materializing logits.

    Replaces: logits = lm_head(h); loss = CE(logits, labels)
    With:     loss = fused_lm_head(h, labels, weights)
    """

    def __init__(
        self,
        tied_weights: SparseTiedWeights,
        v_chunk: int = 2048,
    ):
        super().__init__()
        self.tied_weights = tied_weights
        self.v_chunk = v_chunk

    @property
    def weight(self):
        """For compatibility with code that accesses lm_head.weight"""
        return self.tied_weights.weight

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, S, D] or [N, D]
        labels: Optional[torch.Tensor] = None,          # [B, S] or [N]
        loss_weights: Optional[torch.Tensor] = None,    # [B, S] or [N]
    ) -> torch.Tensor:
        """
        Forward pass - returns loss if labels provided, logits otherwise.
        
        Args:
            hidden_states: Model hidden states
            labels: Target token IDs (optional - if None, returns logits)
            loss_weights: Per-token loss weights (optional - defaults to 1.0)
        
        Returns:
            If labels is None: logits [B, S, V] (WARNING: large tensor!)
            If labels provided: scalar loss
        """
        # Inference mode - return logits
        if labels is None:
            return self.get_logits(hidden_states)
        
        # Flatten to [N, D]
        h_flat = hidden_states.view(-1, hidden_states.shape[-1])
        labels_flat = labels.view(-1)
        weights_flat = loss_weights.view(-1)


        loss = FusedSparseCEFunction.apply(
            h_flat,
            self.tied_weights.weight, # [V, D]
            self.tied_weights.trainable_buffer,
            labels_flat,
            weights_flat,
            self.tied_weights.trainable_start_idx,
            self.v_chunk,
        )

        return loss

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get full logits (for inference/generation).
        WARNING: This materializes full [B, S, V] tensor!
        """
        return hidden_states @ self.tied_weights.weight.T




# ===================
# SPARSE EMBEDDING FUNCTION
# ===================

class SparseEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trainable_buffer, input_ids, tied_weights):
        trainable_mask = (input_ids >= tied_weights.trainable_start_idx)
        ctx.save_for_backward(trainable_buffer, input_ids, trainable_mask)
        ctx.trainable_start_idx = tied_weights.trainable_start_idx
        return tied_weights.weight[input_ids]

    @staticmethod
    def backward(ctx, grad_output):
        trainable_buffer, input_ids, trainable_mask = ctx.saved_tensors
        grad_output = grad_output.float()

        grad = torch.zeros_like(trainable_buffer, dtype=grad_output.dtype)
        trainable_ids = input_ids[trainable_mask] - ctx.trainable_start_idx
        grad.index_add_(0, trainable_ids, grad_output[trainable_mask])

        return grad, None, None

class SparseEmbedding(nn.Module):
    def __init__(self, tied_weights: SparseTiedWeights):
        super().__init__()
        self.tied_weights = tied_weights

    @property
    def weight(self):
        return self.tied_weights.weight

    def forward(self, input_ids):
        return SparseEmbeddingFunction.apply(
            self.tied_weights.trainable_buffer,
            input_ids,
            self.tied_weights
        )


# =============================================================================
# MODEL PATCHING FUNCTION
# =============================================================================


def apply_trainable_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    new_tokens: List[str],
    tie_weights: bool = False,
    v_chunk: int = 2048,
) -> AutoModelForCausalLM:
    """
    Patch model with fused sparse embeddings and LM head.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Tokenizer with new tokens already added
        new_tokens: List of new token strings
        tie_weights: Whether to tie embedding and LM head weights
        v_chunk: Chunk size for fused CE (2048 works well)

    Returns:
        Patched model with memory-efficient fused CE
    """
    # Get original layers
    orig_emb = model.base_model.model.model.embed_tokens
    orig_lm_head = model.base_model.model.lm_head

    trainable_start_idx = len(tokenizer) - len(new_tokens)

    # Create sparse tied weights
    new_emb_weights = SparseTiedWeights(
        weight=orig_emb.weight,
        trainable_start_idx=trainable_start_idx
    )

    new_lm_head_weights = new_emb_weights if tie_weights else SparseTiedWeights(
        weight=orig_lm_head.weight,
        trainable_start_idx=trainable_start_idx
    )

    # Create new modules
    new_emb = SparseEmbedding(new_emb_weights)
    new_lm_head = FusedSparseLMHead(
        new_lm_head_weights,
        v_chunk=v_chunk,
    )

    # Patch model
    model.base_model.model.model.embed_tokens = new_emb
    model.base_model.model.lm_head = new_lm_head

    # Store reference for trainer
    model._fused_lm_head = new_lm_head

    return model


