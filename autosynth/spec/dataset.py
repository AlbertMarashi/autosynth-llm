import torch
from typing import Dict, List
from transformers import AutoTokenizer, DataCollatorWithFlattening
from torch.utils.data import Dataset
from autosynth.spec.serialize import serialise_messages
from autosynth.spec.tokens import MEM_TOKENS, SPECIAL_TOKENS

class CustomDataset(Dataset):
    def __init__(self, conversations: List[List[Dict]], tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        formatted_text = serialise_messages(self.conversations[idx])

        encoding = self.tokenizer(
            formatted_text,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids.tolist(),
            'labels': labels.tolist(),
        }


class CustomDataCollatorWithFlattening(DataCollatorWithFlattening):
    def __init__(self, tokenizer, *args, mem_token_ids_set=None, tok_name_to_id=None, tok_id_to_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.mem_token_ids_set = mem_token_ids_set or set(tokenizer.encode(token, add_special_tokens=False)[0] for token in MEM_TOKENS)
        self.tok_name_to_id = tok_name_to_id or {key: tokenizer.encode(token, add_special_tokens=False)[0] for key, token in SPECIAL_TOKENS.items()}
        self.tok_id_to_name = tok_id_to_name or {v: k for k, v in self.tok_name_to_id.items()}

    def _generate_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        current_loss_score = 0.0
        last_special_token = None

        for i, token_id in enumerate(input_ids.tolist()):
            if token_id in self.mem_token_ids_set:
                loss_mask[i] = 0
                continue
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

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        loss_multiplier = self._generate_loss_mask(batch["input_ids"][0])  # Shape: (total_tokens,)
        batch["loss_multiplier"] = loss_multiplier
        return batch
