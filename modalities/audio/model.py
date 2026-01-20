import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioForensicsModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            use_safetensors=True
        )

        # Freeze encoder (important)
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x: [B, T] waveform
        """
        out = self.encoder(x).last_hidden_state
        pooled = out.mean(dim=1)     # temporal pooling
        return self.head(pooled)



