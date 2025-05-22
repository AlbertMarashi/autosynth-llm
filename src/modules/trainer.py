import torch.nn.functional as F
from transformers.models.auto import AutoModelForCausalLM
from typing import Dict
from transformers import Trainer
# from trl import SFTTrainer  # <-- you're using HuggingFace TRL, not Unsloth's internal engine

class MMSWeightedLossTrainer(Trainer):
    def compute_loss(self, model: AutoModelForCausalLM, inputs: Dict, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        loss_multiplier = inputs["loss_multiplier"]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none"
        )

        # Apply loss multiplier (same shape as labels)
        loss = loss * loss_multiplier.view(-1)
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss
    

