from typing import Sequence

import torch
import torch.nn as nn


def _ensure_tuple(hidden_dim, hidden_dims):
    if hidden_dims is not None:
        if isinstance(hidden_dims, Sequence) and not isinstance(hidden_dims, (str, bytes)):
            return tuple(hidden_dims)
        raise TypeError("hidden_dims must be a sequence of integers.")
    if isinstance(hidden_dim, int):
        return (hidden_dim,)
    if isinstance(hidden_dim, Sequence) and not isinstance(hidden_dim, (str, bytes)):
        return tuple(hidden_dim)
    raise TypeError("hidden_dim must be an int or sequence of ints.")


def _concat_features(u_emb, i_emb, u_feats=None, i_feats=None):
    parts = [u_emb, i_emb]
    if u_feats is not None and u_feats.numel() > 0:
        parts.append(u_feats)
    if i_feats is not None and i_feats.numel() > 0:
        parts.append(i_feats)
    return torch.cat(parts, dim=-1)


class RankerMLP(nn.Module):
    """
    Ranker that uses recall embeddings + side features (already tensors).
    """

    def __init__(
        self,
        user_dim,
        item_dim,
        user_feat_dim=0,
        item_feat_dim=0,
        hidden_dim=128,
        hidden_dims=None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # total input = embeddings + side features
        self.input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim
        self.hidden_dims = _ensure_tuple(hidden_dim, hidden_dims)

        layers = []
        prev_dim = self.input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(prev_dim, 1)

        self.requires_history = False

    def forward(self, u_emb, i_emb, u_feats=None, i_feats=None):
        x = _concat_features(u_emb, i_emb, u_feats, i_feats)
        hidden = self.feature_extractor(x)
        logits = self.output_layer(hidden)
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
        x = _concat_features(u_emb, i_emb, u_feats, i_feats)
        return self.feature_extractor(x)

    @property
    def feature_dim(self) -> int:
        return self.output_layer.in_features
