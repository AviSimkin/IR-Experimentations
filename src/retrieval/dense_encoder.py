"""Dense encoder helpers backed by Hugging Face transformers."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from config.settings import get_torch_device


class DenseEncoder:
    """Thin wrapper around a Hugging Face encoder model."""

    def __init__(self, model_name: str) -> None:
        self.device = get_torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str) -> Any:
        """Encode a string into dense token representations."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]

