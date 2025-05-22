import torch.nn.functional as F
from transformers.models.auto import AutoModelForCausalLM
from typing import Dict
from trl import SFTTrainer
# from cut_cross_entropy.transformers import cce_patch

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
            labels=inputs["labels"],
            # Required for packed sequences
            position_ids=inputs["position_ids"],
            use_cache=False
        )
        per_token_loss = outputs.loss  # Shape: (total_tokens - 1,), assuming reduction='none'
        weights = inputs["loss_multiplier"][1:]  # Align with targets (labels[1:])
        weighted_loss = per_token_loss * weights
        loss = weighted_loss.mean()
        return (loss, outputs) if return_outputs else loss

