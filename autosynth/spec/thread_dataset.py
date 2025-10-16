from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from autosynth.spec.serialize import serialise_messages
from autosynth.spec.tokens import SPECIAL_TOKENS


class ThreadDataset(Dataset):
    """
    Simple dataset that yields individual threads.
    Each thread is tokenized independently.
    """
    
    def __init__(self, conversations: List[List[Dict]], tokenizer: AutoTokenizer):
        """
        Args:
            conversations: List of threads, where each thread is a list of message dicts
            tokenizer: HuggingFace tokenizer
        """
        self.conversations = conversations
        self.tokenizer = tokenizer

        self.tok_name_to_id = {key: tokenizer.encode(token, add_special_tokens=False)[0] for key, token in SPECIAL_TOKENS.items()}
        self.tok_id_to_name = {v: k for k, v in self.tok_name_to_id.items()}


    
    def __len__(self) -> int:
        return len(self.conversations)


    def _generate_loss_mask(self, tokens_ids: List[int]) -> torch.Tensor:
        loss_mask = [1.0] * len(tokens_ids)
        current_loss_score = 0.0
        last_special_token = None

        for i, token_id in enumerate(tokens_ids):
            if token_id not in self.tok_id_to_name:
                loss_mask[i] = current_loss_score
                continue
            loss_mask[i] = 2.0
            if token_id == self.tok_name_to_id["MODEL_TOKEN"]:
                current_loss_score = 1.0
            elif token_id == self.tok_name_to_id["DEVELOPER_TOKEN"]:
                current_loss_score = 0.15
            elif token_id == self.tok_name_to_id["USER_TOKEN"]:
                current_loss_score = 0.5
            elif token_id == self.tok_name_to_id["CONTEXT_TOKEN"]:
                current_loss_score = 0.15
            elif token_id == self.tok_name_to_id["PLATFORM_TOKEN"]:
                current_loss_score = 0.15
            last_special_token = token_id
        return loss_mask

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        Returns:
            Dict with 'input_ids' and 'labels' (both are the same for causal LM)
        """
        thread = self.conversations[idx]
        text = serialise_messages(thread)
        
        # Tokenize without special tokens (we'll handle them in serialise_messages)
        tokens = self.tokenizer(text, add_special_tokens=True)['input_ids']

        loss_multiplier = self._generate_loss_mask(tokens)

        return {
            'input_ids': tokens,
            'labels': tokens,  # Labels = inputs for causal LM
            'loss_multiplier': loss_multiplier,
        }

