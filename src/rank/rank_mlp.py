import torch
import torch.nn as nn

class RankerMLP(nn.Module):
    """
    Ranker that uses recall embeddings + side features (already tensors).
    """

    def __init__(self, user_dim, item_dim,
                 user_feat_dim=0, item_feat_dim=0,
                 hidden_dim=128):
        super().__init__()

        # total input = embeddings + side features
        input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, u_emb, i_emb, u_feats=None, i_feats=None):
        feats = [u_emb, i_emb]

        if u_feats is not None and u_feats.numel() > 0:
            feats.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            feats.append(i_feats)

        x = torch.cat(feats, dim=-1)   # [B, total_dim]
        return self.mlp(x).squeeze(-1) # [B]