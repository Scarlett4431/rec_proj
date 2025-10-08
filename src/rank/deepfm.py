from typing import Sequence

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """DeepFM model that accepts separate feature blocks."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        fm_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.requires_history = False
        input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim
        self.fm_dim = fm_dim
        self.hidden_dims = tuple(hidden_dims)
        self.deep_hidden_dim = self.hidden_dims[-1] if self.hidden_dims else input_dim

        self.linear = nn.Linear(input_dim, 1)
        self.factor = nn.Parameter(torch.randn(input_dim, fm_dim) * 0.01)

        layers = []
        prev_dim = input_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.deep_features = nn.Sequential(*layers) if layers else nn.Identity()
        self.deep_output = nn.Linear(prev_dim, 1)
        self.feature_dim = 1 + fm_dim + prev_dim

    def _concat_features(self, u_emb, i_emb, u_feats=None, i_feats=None):
        parts = [u_emb, i_emb]
        if u_feats is not None and u_feats.numel() > 0:
            parts.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            parts.append(i_feats)
        return torch.cat(parts, dim=-1)

    def _compute_components(self, u_emb, i_emb, u_feats=None, i_feats=None):
        x = self._concat_features(u_emb, i_emb, u_feats, i_feats)
        linear_part = self.linear(x)
        fv = torch.matmul(x, self.factor)
        deep_hidden = self.deep_features(x)
        return x, linear_part, fv, deep_hidden

    def forward(self, u_emb, i_emb, u_feats=None, i_feats=None):
        x, linear_part, fv, deep_hidden = self._compute_components(u_emb, i_emb, u_feats, i_feats)
        fm_part = 0.5 * (fv.pow(2) - torch.matmul(x.pow(2), self.factor.pow(2))).sum(dim=1, keepdim=True)
        deep_part = self.deep_output(deep_hidden)
        logits = linear_part + fm_part + deep_part
        return torch.sigmoid(logits).squeeze(-1)

    def extract_features(
        self,
        u_emb,
        i_emb,
        u_feats=None,
        i_feats=None,
        hist_emb=None,
        hist_mask=None,
    ):
        _, linear_part, fv, deep_hidden = self._compute_components(u_emb, i_emb, u_feats, i_feats)
        return torch.cat([linear_part, fv, deep_hidden], dim=-1)
