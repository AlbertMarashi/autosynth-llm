from typing import List
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



class SparseTiedWeights(nn.Module):
    def __init__(self, weights, trainable_start_idx):
        """
        A custom class to handle weights that are only partially trainable.
        Args:
            weights (Tensor): Original weights from the embedding layer.
                Shape: (vocab_size, embedding_dim)
            trainable_start_idx (int): Index where trainable weights start.
        """
        super().__init__()
        self._weight = weights  # Original embedding weights tensor, Shape: (vocab_size, embedding_dim)
        self.trainable_start_idx = trainable_start_idx  # Index where the trainable portion begins

        # Store trainable weights in a separate temporary buffer for updating
        self.trainable_buffer = nn.Parameter(
            self._weight[trainable_start_idx:].clone().float()  # Shape: (num_trainable_tokens, embedding_dim)
        )

    @property
    def weight(self):
        """
        Return the combined weights with trainable values updated from the buffer.
        Returns:
            weight (Tensor): Combined weights of the embedding layer.
                Shape: (vocab_size, embedding_dim)
        """
        # LLaMA expects this
        with torch.no_grad():
            self._weight[self.trainable_start_idx:] = self.trainable_buffer
        return self._weight

    def set_weight(self, new_weights):
        """
        Set new weights for the entire embedding layer.
        Args:
            new_weights (Tensor): New weights tensor.
                Shape: (vocab_size, embedding_dim)
        """
        self._weight = new_weights
        self.trainable_buffer = nn.Parameter(
            self._weight[self.trainable_start_idx:].clone().float()  # Shape: (num_trainable_tokens, embedding_dim)
        )

class SparseEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trainable_buffer, input_ids, tied_weights):
        """
        Custom forward pass for sparse embedding. Selects the appropriate embeddings based on input_ids.

        Args:
            trainable_buffer (Tensor): Buffer containing the trainable weights.
                Shape: (num_trainable_tokens, embedding_dim)
            input_ids (Tensor): Input token ids.
                Shape: (batch_size, seq_length)
            tied_weights (SparseTiedWeights): The tied weights object.

        Returns:
            output (Tensor): Embedding representations for input tokens.
                Shape: (batch_size, seq_length, embedding_dim)
        """
        # Determine which inputs are trainable
        trainable_mask = (input_ids >= tied_weights.trainable_start_idx)  # Shape: (batch_size, seq_length)
        ctx.save_for_backward(trainable_buffer, input_ids, trainable_mask)
        ctx.trainable_start_idx = tied_weights.trainable_start_idx

        return tied_weights.weight[input_ids]  # Shape: (batch_size, seq_length, embedding_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for sparse embedding.

        Args:
            grad_output (Tensor): Gradients from the subsequent layers.
                Shape: (batch_size, seq_length, embedding_dim)

        Returns:
            grad (Tensor): Gradient for the trainable buffer.
                Shape: (num_trainable_tokens, embedding_dim)
        """
        trainable_buffer, input_ids, trainable_mask = ctx.saved_tensors
        grad_output = grad_output.float()  # Shape: (batch_size, seq_length, embedding_dim)

        # Create gradient buffer with zero values
        grad = torch.zeros_like(trainable_buffer, dtype=grad_output.dtype)  # Shape: (num_trainable_tokens, embedding_dim)
        trainable_ids = input_ids[trainable_mask] - ctx.trainable_start_idx  # Shape: (num_trainable_tokens,)
        grad.index_add_(0, trainable_ids, grad_output[trainable_mask])

        return grad, None, None

class SparseEmbedding(nn.Module):
    def __init__(self, tied_weights: SparseTiedWeights):
        """
        Sparse embedding layer for handling tied weights with partial trainability.
        Args:
            tied_weights (SparseTiedWeights): An instance of SparseTiedWeights.
        """
        super().__init__()
        self.tied_weights = tied_weights

    @property
    def weight(self):
        """
        Retrieve the current weights of the embedding layer.
        Returns:
            weight (Tensor): The complete weights tensor.
                Shape: (vocab_size, embedding_dim)
        """
        return self.tied_weights.weight

    def forward(self, input_ids):
        """
        Forward pass for sparse embedding layer.

        Args:
            input_ids (Tensor): Input token ids.
                Shape: (batch_size, seq_length)

        Returns:
            output (Tensor): Embedding representations for input tokens.
                Shape: (batch_size, seq_length, embedding_dim)
        """
        return SparseEmbeddingFunction.apply(
            self.tied_weights.trainable_buffer,  # Shape: (num_trainable_tokens, embedding_dim)
            input_ids,  # Shape: (batch_size, seq_length)
            self.tied_weights
        )

class SparseLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trainable_buffer, x, tied_weights):
        """
        Forward pass for the sparse language model head.

        Args:
            trainable_buffer (Tensor): Buffer containing trainable weights for LM head.
                Shape: (num_trainable_tokens, embedding_dim)
            x (Tensor): Input features from the decoder.
                Shape: (batch_size, seq_length, embedding_dim)
            tied_weights (SparseTiedWeights): The tied weights object.

        Returns:
            output (Tensor): Output logits.
                Shape: (batch_size, seq_length, vocab_size)
        """
        ctx.weight = tied_weights.weight  # Shape: (vocab_size, embedding_dim)
        ctx.save_for_backward(trainable_buffer, x)
        ctx.trainable_start_idx = tied_weights.trainable_start_idx

        return x @ tied_weights.weight.t()  # Shape: (batch_size, seq_length, vocab_size)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the sparse language model head.

        Args:
            grad_output (Tensor): Gradients from subsequent layers.
                Shape: (batch_size, seq_length, vocab_size)

        Returns:
            grad_buffer (Tensor): Gradient for trainable buffer.
                Shape: (num_trainable_tokens, embedding_dim)
            grad_input_x (Tensor): Gradient for the input tensor x.
                Shape: (batch_size, seq_length, embedding_dim)
        """
        trainable_buffer, x = ctx.saved_tensors

        grad_input_x = grad_output @ ctx.weight  # Shape: (batch_size, seq_length, embedding_dim)
        trainable_grads = grad_output[..., ctx.trainable_start_idx:]  # Shape: (batch_size, seq_length, num_trainable_tokens)

        grad_buffer = (
            trainable_grads.permute(2, 0, 1)  # Shape: (num_trainable_tokens, batch_size, seq_length)
                .reshape(trainable_grads.size(-1), -1)  # (V_t, B*S) - no copy, just view
            @ x.reshape(-1, x.size(-1))  # Shape: (batch_size * seq_length, embedding_dim)
        )  # Shape: (num_trainable_tokens, embedding_dim) ← Only storing gradients for trainable tokens!

        return grad_buffer, grad_input_x, None

class SparseLMHead(nn.Module):
    def __init__(self, tied_weights: SparseTiedWeights):
        """
        Sparse language model head that handles tied weights with partial trainability.
        Args:
            tied_weights (SparseTiedWeights): An instance of SparseTiedWeights.
        """
        super().__init__()
        self.tied_weights = tied_weights

    @property
    def weight(self):
        """
        Retrieve the current weights of the LM head.
        Returns:
            weight (Tensor): The complete weights tensor.
                Shape: (vocab_size, embedding_dim)
        """
        return self.tied_weights.weight

    def forward(self, x):
        """
        Forward pass for sparse LM head.

        Args:
            x (Tensor): Input features from the decoder.
                Shape: (batch_size, seq_length, embedding_dim)

        Returns:
            output (Tensor): Output logits.
                Shape: (batch_size, seq_length, vocab_size)
        """
        return SparseLMHeadFunction.apply(
            self.tied_weights.trainable_buffer,  # Shape: (num_trainable_tokens, embedding_dim)
            x,  # Shape: (batch_size, seq_length, embedding_dim)
            self.tied_weights
        )



def apply_trainable_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    new_tokens: List[str],
    tie_weights: bool = False,
):
    orig_emb = model.base_model.model.model.embed_tokens
    orig_lm_head = model.base_model.model.lm_head

    new_emb_weights = SparseTiedWeights(
        weights=orig_emb.weight,
        trainable_start_idx=len(tokenizer) - len(new_tokens)
    )

    new_lm_head_weights = new_emb_weights if tie_weights else SparseTiedWeights(
        weights=orig_lm_head.weight,  # Shape: (vocab_size, embedding_dim)
        trainable_start_idx=len(tokenizer) - len(new_tokens)
    )

    new_emb = SparseEmbedding(new_emb_weights)
    new_lm_head = SparseLMHead(new_emb_weights) if tie_weights else SparseLMHead(new_lm_head_weights)

    model.base_model.model.model.embed_tokens = new_emb
    model.base_model.model.lm_head = new_lm_head

    return model

# # Instantiate SparseTiedWeights using original embedding weights
# # We assume `orig_emb.weight` is the original embedding weights of the model.
# tie_weights = False

# new_emb_weights = SparseTiedWeights(
#     weights=orig_emb.weight,  # Shape: (vocab_size, embedding_dim)
#     trainable_start_idx=len(tokenizer) - len(new_tokens)
# )

# new_lm_head_weights = new_emb_weights if tie_weights else SparseTiedWeights(
#     weights=orig_lm_head.weight,  # Shape: (vocab_size, embedding_dim)
#     trainable_start_idx=len(tokenizer) - len(new_tokens)
# )

# # Create new instances of the sparse embedding and LM head layers
# new_emb = SparseEmbedding(new_emb_weights)
# new_lm_head = SparseLMHead(new_lm_head_weights)

# # Replace the existing model's embedding and LM head layers with the sparse ones
# model.base_model.model.model.embed_tokens = new_emb
# model.base_model.model.lm_head = new_lm_head


