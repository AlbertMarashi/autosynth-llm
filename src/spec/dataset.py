from typing import Any, Dict, List
from torch.utils.data import Dataset
from src.spec.serialize import serialise_messages
from src.spec.tok_vocab import MEM_TOKENS, SPECIAL_TOKENS
import torch


class CustomDataset(Dataset):
    def __init__(self, conversations: List[List[Dict]], tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = conversations

        self.mem_token_ids_set = set(self.tokenizer.encode(token, add_special_tokens=False)[0] for token in MEM_TOKENS)

        self.tok_name_to_id = {
            key: self.tokenizer.encode(token, add_special_tokens=False)[0]
            for key, token in SPECIAL_TOKENS.items()
        }
        self.tok_id_to_name = {
            v: k for k, v in self.tok_name_to_id.items()
        }


    def _generate_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)

        current_loss_score = 0.0
        last_special_token = None

        for i, token_id in enumerate(input_ids.tolist()):
            # If this is a memory token, we wanna set it's score to MEMORY_TOKEN_LOSS
            if token_id in self.mem_token_ids_set:
                loss_mask[i] = 0
                continue

            # Regular token content, so just use last loss score
            if token_id not in self.tok_id_to_name:
                loss_mask[i] = current_loss_score
                continue

        
            # else, this is a main special token
            # so we wanna set the loss score to 2.0
            loss_mask[i] = 2.0
            if token_id == self.tok_name_to_id["FORMAT"]:
                # format tokens should be scored highly,
                # so the model learns to predict the correct format
                current_loss_score = 2.0
            elif token_id == self.tok_name_to_id["CONTENT"]:
                # if we ran into a content token,
                # and the last content type was a format token,
                # then it means we're about to start the content of the assistant message
                if last_special_token == self.tok_name_to_id["FORMAT"]:
                    current_loss_score = 1.0
            elif token_id == self.tok_name_to_id["DEVELOPER_TOKEN"]:
                current_loss_score = 0.3
            elif token_id == self.tok_name_to_id["USER_TOKEN"]:
                current_loss_score = 0.4
            elif token_id == self.tok_name_to_id["CONTEXT_TOKEN"]:
                current_loss_score = 0.3
            elif token_id == self.tok_name_to_id["PLATFORM_TOKEN"]:
                current_loss_score = 0.4

            # Set the last special token to the special token we just ran into
            last_special_token = token_id
        return loss_mask

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        formatted_text = serialise_messages(self.conversations[idx])
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        loss_multiplier = self._generate_loss_mask(input_ids)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'loss_multiplier': loss_multiplier,
            'attention_mask': attention_mask
        }
    
