from typing import List, Dict
import torch
from transformers import AutoTokenizer

class PackedCollator:
    """
    Packs multiple threads into a single sequence up to max_seq_length.

    - Resets position_ids at thread boundaries
    - Generates loss multipliers based on special tokens
    - No padding (variable length batches)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum tokens per packed batch
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.cat([feature['input_ids'] for feature in features])[:self.max_seq_length]
        labels = torch.cat([feature['labels'] for feature in features])[:self.max_seq_length]
        position_ids = torch.cat([feature['position_ids'] for feature in features])[:self.max_seq_length]
        loss_multiplier = torch.cat([feature['loss_multiplier'] for feature in features])[:self.max_seq_length]
        return {
            'input_ids': input_ids.unsqueeze(0).detach(),
            'labels': labels.unsqueeze(0).detach(),
            'position_ids': position_ids.unsqueeze(0).detach(),
            'loss_multiplier': loss_multiplier.unsqueeze(0).detach()
        }
