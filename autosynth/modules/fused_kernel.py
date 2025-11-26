"""
Benchmark: Fused Sparse Cross-Entropy (Fixed)

Recomputes label indices in backward (matching original structure).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def chunked_ce_forward(
    h: torch.Tensor, # [N, D] hidden states
    W: torch.Tensor, # [V, D] weight matrix (NOT transposed)
    labels: torch.Tensor, # [N] target indices
    loss_weights: torch.Tensor, # [N] per-token weights
    V_CHUNK: int = 2048, # chunk size for the weight matrix
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass with online logsumexp.
    """
    N, D = h.shape
    V = W.shape[0]
    device = h.device
    
    h_fp16 = h if h.dtype == torch.float16 else h.half()
    
    # Accumulators - only running_sum needs FP32
    running_max = torch.full((N,), -float('inf'), dtype=torch.float16, device=device)
    running_sum = torch.zeros(N, dtype=torch.float32, device=device)
    label_logits = torch.zeros(N, dtype=torch.float16, device=device)
    
    # Precompute label indices
    safe_labels = labels.clamp(min=0, max=V-1)
    label_chunk_idx = safe_labels // V_CHUNK
    label_local_idx = safe_labels % V_CHUNK
    
    num_chunks = (V + V_CHUNK - 1) // V_CHUNK
    
    for chunk_idx in range(num_chunks):
        v_start = chunk_idx * V_CHUNK
        v_end = min(v_start + V_CHUNK, V)
        V_cs = v_end - v_start
        
        logits_chunk = torch.mm(h_fp16, W[v_start:v_end].T)
        
        # Online LSE
        chunk_max = logits_chunk.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp((running_max - new_max).float())
        new_contrib = torch.exp(logits_chunk - new_max.unsqueeze(1)).sum(dim=1).float()
        running_sum = old_scale * running_sum + new_contrib
        running_max = new_max
        
        # Vectorized label extraction
        in_chunk = (label_chunk_idx == chunk_idx)
        local_idx = label_local_idx.clamp(0, V_cs - 1)
        gathered = logits_chunk.gather(1, local_idx.unsqueeze(1)).squeeze(1)
        label_logits = torch.where(in_chunk, gathered, label_logits)
    
    lse = running_max.float() + torch.log(running_sum)
    weight_sum = loss_weights.sum()
    loss = (loss_weights * (-label_logits.float() + lse)).sum() / weight_sum.clamp(min=1e-8)
    
    return loss, lse, weight_sum  # Only 3 returns like original


# =============================================================================
# BACKWARD
# =============================================================================

def chunked_ce_backward(
    h: torch.Tensor,            # [N, D] hidden states
    W: torch.Tensor,            # [V, D] (NOT transposed)
    labels: torch.Tensor,       # [N] target indices
    loss_weights: torch.Tensor, # [N] per-token weights
    lse: torch.Tensor,          # [N] log-sum-exp
    weight_sum: torch.Tensor,   # scalar
    trainable_start_idx: int,   # index of the first trainable token
    V_CHUNK: int = 2048,        # chunk size for the weight matrix
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass with FP16 accumulation.
    """
    N, D = h.shape
    V = W.shape[0]
    V_t = V - trainable_start_idx
    device = h.device
    
    h_fp16 = h if h.dtype == torch.float16 else h.half()
    norm_weights = loss_weights / weight_sum.clamp(min=1e-8)
    
    grad_h = torch.zeros(N, D, dtype=torch.float16, device=device)
    grad_W_trainable = torch.zeros(V_t, D, dtype=torch.float16, device=device)
    
    # Recompute label indices (cheap, avoids saving)
    safe_labels = labels.clamp(min=0, max=V-1)
    label_chunk_idx = safe_labels // V_CHUNK
    label_local_idx = safe_labels % V_CHUNK
    
    num_chunks = (V + V_CHUNK - 1) // V_CHUNK
    
    for chunk_idx in range(num_chunks):
        v_start = chunk_idx * V_CHUNK
        v_end = min(v_start + V_CHUNK, V)
        V_cs = v_end - v_start
        
        W_chunk = W[v_start:v_end]
        logits_chunk = torch.mm(h_fp16, W_chunk.T)
        
        # Softmax from saved LSE
        softmax_chunk = torch.exp(logits_chunk.float() - lse.unsqueeze(1))
        grad_logits = (norm_weights.unsqueeze(1) * softmax_chunk).half()
        
        # Vectorized label subtraction (no torch.where, just multiply by mask)
        in_chunk = (label_chunk_idx == chunk_idx)
        local_idx = label_local_idx.clamp(0, V_cs - 1)
        subtract_vals = in_chunk.half() * norm_weights.half()  # 0 if not in chunk
        grad_logits.scatter_add_(1, local_idx.unsqueeze(1), -subtract_vals.unsqueeze(1))
        
        # Accumulate
        grad_h = torch.addmm(grad_h, grad_logits, W_chunk)
        
        if v_end > trainable_start_idx:
            t_start_local = max(0, trainable_start_idx - v_start)
            t_start_global = max(0, v_start - trainable_start_idx)
            t_size = V_cs - t_start_local
            grad_W_trainable[t_start_global:t_start_global + t_size] = torch.addmm(
                grad_W_trainable[t_start_global:t_start_global + t_size],
                grad_logits[:, t_start_local:].T, h_fp16
            )
    
    return grad_h, grad_W_trainable

