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
        """
        Pack threads from features into a single batch.
        
        Args:
            features: List of dicts with 'input_ids' and 'labels' from dataset
        
        Returns:
            Dict with:
                - input_ids: [1, seq_len]
                - labels: [1, seq_len]
                - position_ids: [1, seq_len]
                - loss_multiplier: [1, seq_len]
        """
        packed = self._pack_threads(features)
        
        # Convert to tensors with batch dimension
        return {
            'input_ids': torch.tensor([packed['input_ids']], dtype=torch.long),
            'labels': torch.tensor([packed['labels']], dtype=torch.long),
            'position_ids': torch.tensor([packed['positions']], dtype=torch.long),
            'loss_multiplier': torch.tensor([packed['loss_multiplier']], dtype=torch.float32),
        }
    
    def _pack_threads(self, features: List[Dict]) -> Dict[str, List]:
        """Core packing logic."""
        packed_input_ids = []
        packed_labels = []
        packed_positions = []
        packed_loss_multiplier = []
        
        current_position = 0
        
        for feature in features:
            thread_ids = feature['input_ids']
            
            # Calculate space needed (thread)
            total_needed = len(packed_input_ids) + len(thread_ids)
            
            # Stop if thread doesn't fit
            if total_needed > self.max_seq_length:
                break
            
            # Add thread
            packed_input_ids.extend(thread_ids)
            packed_labels.extend(thread_ids)
            
            # Generate position IDs (reset at thread boundaries)
            thread_positions = list(range(current_position, current_position + len(thread_ids)))
            packed_positions.extend(thread_positions)
            
            # Generate loss weights
            packed_loss_multiplier.extend(feature['loss_multiplier'])
            
            # Update position counter
            current_position += len(thread_ids)
        
        return {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'positions': packed_positions,
            'loss_multiplier': packed_loss_multiplier,
        }
