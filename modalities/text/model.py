import torch.nn as nn
from transformers import AutoModel


class TextForensicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:,0]
        return self.head(cls).squeeze(-1)




