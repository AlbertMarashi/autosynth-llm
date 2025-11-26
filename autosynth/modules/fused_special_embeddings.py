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
from typing import Optional, Tuple, List
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CORE FUSED OPERATIONS
# =============================================================================

def chunked_ce_forward(
    h: torch.Tensor,            # [N, D] hidden states
    W: torch.Tensor,            # [V, D] weight matrix (NOT transposed)
    labels: torch.Tensor,       # [N] target indices
    loss_weights: torch.Tensor, # [N] per-token weights
    ignore_index: int = -100,
    V_CHUNK: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass with online logsumexp.

    Uses W.T directly - cuBLAS handles transpose efficiently via transB flag.
    """
    N, D = h.shape
    V = W.shape[0]
    device = h.device

    h_fp16 = h.half() if h.dtype != torch.float16 else h

    valid_mask = (labels != ignore_index)

    # FP32 accumulators for numerical stability
    running_max = torch.full((N,), -float('inf'), dtype=torch.float32, device=device)
    running_sum = torch.zeros(N, dtype=torch.float32, device=device)
    label_logits = torch.zeros(N, dtype=torch.float32, device=device)

    safe_labels = labels.clamp(min=0)
    label_chunks = safe_labels // V_CHUNK
    label_local_idx = safe_labels % V_CHUNK

    num_chunks = (V + V_CHUNK - 1) // V_CHUNK

    for chunk_idx in range(num_chunks):
        v_start = chunk_idx * V_CHUNK
        v_end = min(v_start + V_CHUNK, V)

        # Slice rows of W, then transpose for GEMM
        # W[v_start:v_end] is [V_chunk, D] contiguous
        # .T gives [D, V_chunk] view - cuBLAS handles this efficiently
        W_chunk = W[v_start:v_end]  # [V_chunk, D]

        # FP16 GEMM: [N, D] @ [D, V_chunk] -> [N, V_chunk]
        logits_chunk = torch.mm(h_fp16, W_chunk.T)
        logits_f32 = logits_chunk.float()

        # Online LSE update
        chunk_max = logits_f32.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        new_contrib = torch.exp(logits_f32 - new_max.unsqueeze(1)).sum(dim=1)
        running_sum = old_scale * running_sum + new_contrib
        running_max = new_max

        # Extract label logits
        labels_in_chunk = (label_chunks == chunk_idx) & valid_mask
        if labels_in_chunk.any():
            row_idx = torch.where(labels_in_chunk)[0]
            col_idx = label_local_idx[labels_in_chunk]
            label_logits[labels_in_chunk] = logits_f32[row_idx, col_idx]

    lse = running_max + torch.log(running_sum)

    loss_weights_f32 = loss_weights.float()
    valid_weights = loss_weights_f32 * valid_mask.float()
    weight_sum = valid_weights.sum()

    per_token_loss = valid_weights * (-label_logits + lse)
    loss = per_token_loss.sum() / weight_sum.clamp(min=1e-8)

    return loss, lse, weight_sum, valid_mask


def chunked_ce_backward(
    h: torch.Tensor,            # [N, D]
    W: torch.Tensor,            # [V, D] (NOT transposed)
    labels: torch.Tensor,       # [N]
    loss_weights: torch.Tensor, # [N]
    lse: torch.Tensor,          # [N]
    weight_sum: torch.Tensor,   # scalar
    valid_mask: torch.Tensor,   # [N]
    trainable_start_idx: int,
    ignore_index: int = -100,
    V_CHUNK: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass with FP16 accumulation.
    """
    N, D = h.shape
    V = W.shape[0]
    V_t = V - trainable_start_idx
    device = h.device

    h_fp16 = h.half() if h.dtype != torch.float16 else h

    loss_weights_f32 = loss_weights.float() * valid_mask.float()
    norm_weights = loss_weights_f32 / weight_sum.float().clamp(min=1e-8)

    grad_h = torch.zeros(N, D, dtype=torch.float16, device=device)
    grad_W_trainable = torch.zeros(V_t, D, dtype=torch.float16, device=device)

    safe_labels = labels.clamp(min=0)
    label_chunks = safe_labels // V_CHUNK
    label_local_idx = safe_labels % V_CHUNK

    num_chunks = (V + V_CHUNK - 1) // V_CHUNK

    for chunk_idx in range(num_chunks):
        v_start = chunk_idx * V_CHUNK
        v_end = min(v_start + V_CHUNK, V)
        V_chunk_size = v_end - v_start

        W_chunk = W[v_start:v_end]  # [V_chunk, D]

        # Recompute logits
        logits_chunk = torch.mm(h_fp16, W_chunk.T)

        # Softmax from saved LSE
        softmax_chunk = torch.exp(logits_chunk.float() - lse.unsqueeze(1))
        grad_logits = norm_weights.unsqueeze(1) * softmax_chunk

        # Subtract at label positions
        labels_in_chunk = (label_chunks == chunk_idx) & valid_mask
        if labels_in_chunk.any():
            row_idx = torch.where(labels_in_chunk)[0]
            col_idx = label_local_idx[labels_in_chunk]
            grad_logits[row_idx, col_idx] -= norm_weights[labels_in_chunk]

        grad_logits_fp16 = grad_logits.half()

        # grad_h += grad_logits @ W_chunk
        # [N, V_chunk] @ [V_chunk, D] -> [N, D]
        grad_h = torch.addmm(grad_h, grad_logits_fp16, W_chunk)

        # grad_W for trainable portion
        if v_end > trainable_start_idx:
            train_start_local = max(0, trainable_start_idx - v_start)
            train_end_local = V_chunk_size

            train_start_global = max(0, v_start - trainable_start_idx)
            train_end_global = train_start_global + (train_end_local - train_start_local)

            grad_logits_trainable = grad_logits_fp16[:, train_start_local:train_end_local]

            # [V_chunk_t, N] @ [N, D] -> [V_chunk_t, D]
            grad_W_trainable[train_start_global:train_end_global] = torch.addmm(
                grad_W_trainable[train_start_global:train_end_global],
                grad_logits_trainable.T,
                h_fp16
            )

    return grad_h, grad_W_trainable


# =============================================================================
# AUTOGRAD FUNCTION
# =============================================================================

class FusedSparseCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        W: torch.Tensor,              # [V, D] - NOT transposed
        trainable_buffer: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
        trainable_start_idx: int,
        ignore_index: int,
        V_CHUNK: int,
    ) -> torch.Tensor:

        loss, lse, weight_sum, valid_mask = chunked_ce_forward(
            h, W, labels, loss_weights, ignore_index, V_CHUNK
        )

        weight_sum_t = weight_sum.clone() if isinstance(weight_sum, torch.Tensor) else torch.tensor(weight_sum, device=h.device)

        ctx.save_for_backward(h, W, labels, loss_weights, lse, weight_sum_t, valid_mask)
        ctx.trainable_start_idx = trainable_start_idx
        ctx.ignore_index = ignore_index
        ctx.V_CHUNK = V_CHUNK

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        h, W, labels, loss_weights, lse, weight_sum_t, valid_mask = ctx.saved_tensors

        grad_h, grad_W_trainable = chunked_ce_backward(
            h, W, labels, loss_weights, lse, weight_sum_t, valid_mask,
            ctx.trainable_start_idx, ctx.ignore_index, ctx.V_CHUNK
        )

        grad_h = grad_h * grad_output
        grad_W_trainable = grad_W_trainable * grad_output

        return grad_h, None, grad_W_trainable, None, None, None, None, None


# =============================================================================
# SPARSE TIED WEIGHTS (Compatible with existing code)
# =============================================================================

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


# =============================================================================
# FUSED LM HEAD MODULE
# =============================================================================

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
        ignore_index: int = -100,
    ):
        super().__init__()
        self.tied_weights = tied_weights
        self.v_chunk = v_chunk
        self.ignore_index = ignore_index

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
            self.ignore_index,
            self.v_chunk,
        )

        return loss

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get full logits (for inference/generation).
        WARNING: This materializes full [B, S, V] tensor!
        """
        return hidden_states @ self.tied_weights.weight.T


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
        ignore_index=-100
    )

    # Patch model
    model.base_model.model.model.embed_tokens = new_emb
    model.base_model.model.lm_head = new_lm_head

    # Store reference for trainer
    model._fused_lm_head = new_lm_head

    return model


