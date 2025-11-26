from transformers.models.auto import AutoModelForCausalLM
from typing import Dict
from trl import SFTTrainer

class MMSWeightedLossTrainer(SFTTrainer):
    """
    Trainer that uses fused CE loss for memory efficiency.

    Expects model to have been patched with apply_fused_trainable_embeddings()
    """

    def compute_loss(
        self,
        model: AutoModelForCausalLM,
        inputs: Dict,
        **kwargs,
    ):
        # Sync the weights from previous run...
        model._fused_lm_head.tied_weights.sync()

        # Get hidden states (not logits!)
        outputs = model.model.model(
            input_ids=inputs["input_ids"],
            position_ids=inputs.get("position_ids"),
        )

        hidden_states = outputs.last_hidden_state  # [B, S, D]

        # NO SHIFTING - data is already aligned!
        # hidden_states[i] predicts labels[i]
        loss = model._fused_lm_head(
            hidden_states,           # [B, S, D]
            inputs["labels"],        # [B, S] - already shifted in dataset
            inputs["loss_multiplier"] # [B, S] - already shifted in dataset
        )

        return loss