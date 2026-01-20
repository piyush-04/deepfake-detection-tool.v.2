import torch
import torch.nn as nn

class VideoConsistencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.GRU(1, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.temporal(x)
        return self.head(out[:, -1]).squeeze(-1)



