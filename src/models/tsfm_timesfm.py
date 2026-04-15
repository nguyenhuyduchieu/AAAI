from __future__ import annotations

from typing import Dict

import numpy as np
from transformers import AutoModel


class TimesFMLatentExtractor:
    """TimesFM wrapper for hidden-state extraction.

    Uses Hugging Face model API so we can collect token-level and layer-wise
    embeddings via output_hidden_states=True.
    """

    def __init__(self, model_id: str) -> None:
        self.model = AutoModel.from_pretrained(model_id)

    def extract(self, inputs_embeds) -> Dict[str, np.ndarray]:
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state.detach().cpu().numpy()
        layers = [h.detach().cpu().numpy() for h in outputs.hidden_states]
        seq_emb = last_hidden.mean(axis=1)
        return {
            "token_embeddings": last_hidden,
            "sequence_embeddings": seq_emb,
            "layer_embeddings": np.stack(layers, axis=0),
        }
