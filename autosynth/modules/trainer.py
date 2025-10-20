from transformers.models.auto import AutoModelForCausalLM
from typing import Dict
from trl import SFTTrainer
from torch import nn
import torch.nn.functional as F

class MMSWeightedLossTrainer(SFTTrainer):
    def compute_loss(
            self,
            model: AutoModelForCausalLM,
            inputs: Dict,
            return_outputs=False,
            **kwargs,
        ):
        
        outputs = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
        )

        # Get logits and labels
        logits = outputs.logits  # [batch, seq_len, vocab]
        labels = inputs["labels"]  # [batch, seq_len]

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token loss (this is what you want!)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # [N, vocab]
            shift_labels.view(-1),  # [N]
            reduction='none',  # <-- This gives you per-token losses!
            ignore_index=-100
        )

        weights = inputs["loss_multiplier"][..., 1:].contiguous().view(-1)
        
        # Compute final weighted loss
        weighted_loss = per_token_loss * weights
        loss = weighted_loss.sum() / (weights > 0).sum()
        
        return (loss, outputs) if return_outputs else loss