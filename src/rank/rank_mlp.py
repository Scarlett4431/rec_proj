import torch
import torch.nn as nn

class RankerMLP(nn.Module):
    def __init__(self, user_dim, item_dim, extra_dim=0, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_emb, item_emb, extra_feats=None):
        if extra_feats is not None:
            x = torch.cat([user_emb, item_emb, extra_feats], dim=1)
        else:
            x = torch.cat([user_emb, item_emb], dim=1)
        out = self.mlp(x)           # [B, 1]
        return out.squeeze(-1)      # [B]  (safe even when B=1)