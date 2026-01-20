import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ImageForensicsModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        out = self.encoder(pixel_values=x)
        feat = out.last_hidden_state[:, 0]
        return self.head(feat)



