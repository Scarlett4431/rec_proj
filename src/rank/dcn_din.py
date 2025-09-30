import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rank.dcn import CrossNetwork
from src.rank.din import AttentionUnit


class DCNDINRanker(nn.Module):
    """Combines DIN-style attention over histories with a Deep & Cross tower."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        max_history: int = 50,
        cross_layers: int = 3,
        attention_hidden=(64, 32),
        hidden_dims=(256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.max_history = max_history
        self.requires_history = True

        self.attention = AttentionUnit(item_dim, attention_hidden)

        input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim + item_dim
        self.cross_net = CrossNetwork(input_dim, cross_layers)

        deep_layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden))
            deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.deep = nn.Sequential(*deep_layers) if deep_layers else nn.Identity()

        combined_dim = input_dim + (hidden_dims[-1] if hidden_dims else input_dim)
        self.output_layer = nn.Linear(combined_dim, 1)

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        u_feats: torch.Tensor = None,
        i_feats: torch.Tensor = None,
        hist_emb: torch.Tensor = None,
        hist_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if hist_emb is None or hist_mask is None:
            raise ValueError("DCNDINRanker requires history embeddings and mask")

        att_logits = self.attention(hist_emb, i_emb)
        att_logits = att_logits.masked_fill(~hist_mask, float("-inf"))
        att_weights = F.softmax(att_logits, dim=1)
        att_weights = torch.where(hist_mask, att_weights, torch.zeros_like(att_weights))
        att_weights = torch.nan_to_num(att_weights, nan=0.0)
        hist_rep = torch.sum(att_weights.unsqueeze(-1) * hist_emb, dim=1)
        valid_hist = hist_mask.any(dim=1, keepdim=True)
        hist_rep = torch.where(valid_hist, hist_rep, torch.zeros_like(hist_rep))

        parts = [u_emb, i_emb, hist_rep]
        if u_feats is not None and u_feats.numel() > 0:
            parts.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            parts.append(i_feats)
        x = torch.cat(parts, dim=-1)

        cross_out = self.cross_net(x)
        deep_out = self.deep(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.output_layer(combined)
        return torch.sigmoid(logits).squeeze(-1)
