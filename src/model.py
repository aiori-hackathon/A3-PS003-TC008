import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (batch, 12 hours)
        out, _ = self.gru(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze(-1)   # predicted next hour