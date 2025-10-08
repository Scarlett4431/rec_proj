from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUnit(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        input_dim = embed_dim * 4
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.att_mlp = nn.Sequential(*layers)

    def forward(self, hist_emb, target_emb):
        # hist_emb: [B, L, D], target_emb: [B, D]
        target_exp = target_emb.unsqueeze(1).expand_as(hist_emb)
        concat = torch.cat(
            [hist_emb, target_exp, hist_emb * target_exp, hist_emb - target_exp], dim=-1
        )
        attn = self.att_mlp(concat).squeeze(-1)
        return attn


class DINRanker(nn.Module):
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        embed_dim: int = 64,
        attention_hidden: Sequence[int] = (64, 32),
        dnn_hidden: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.embed_dim = item_dim  # assume provided embeddings already sized appropriately
        self.requires_history = True
        self.attention_hidden = tuple(attention_hidden)
        self.dnn_hidden = tuple(dnn_hidden)

        self.attention = AttentionUnit(item_dim, self.attention_hidden)

        dnn_input_dim = user_dim + item_dim + item_dim + user_feat_dim + item_feat_dim
        layers = []
        prev_dim = dnn_input_dim
        for hidden in self.dnn_hidden:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.dnn_features = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(prev_dim, 1)
        self.feature_dim = prev_dim

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        hist_emb: torch.Tensor,
        hist_mask: torch.Tensor,
        u_feats: torch.Tensor = None,
        i_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        features = self.extract_features(
            u_emb=u_emb,
            i_emb=i_emb,
            hist_emb=hist_emb,
            hist_mask=hist_mask,
            u_feats=u_feats,
            i_feats=i_feats,
        )
        logits = self.output_layer(features)
        return torch.sigmoid(logits).squeeze(-1)

    def _encode_history(self, hist_emb, hist_mask, target_emb):
        if hist_emb is None or hist_mask is None:
            raise ValueError("DINRanker requires history embeddings and mask")
        att_logits = self.attention(hist_emb, target_emb)
        att_logits = att_logits.masked_fill(~hist_mask, float("-inf"))
        att_weights = F.softmax(att_logits, dim=1)
        att_weights = torch.where(hist_mask, att_weights, torch.zeros_like(att_weights))
        att_weights = torch.nan_to_num(att_weights, nan=0.0)
        hist_rep = torch.sum(att_weights.unsqueeze(-1) * hist_emb, dim=1)
        valid_hist = hist_mask.any(dim=1, keepdim=True)
        hist_rep = torch.where(valid_hist, hist_rep, torch.zeros_like(hist_rep))
        return hist_rep

    def extract_features(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        hist_emb: torch.Tensor = None,
        hist_mask: torch.Tensor = None,
        u_feats: torch.Tensor = None,
        i_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        hist_rep = self._encode_history(hist_emb, hist_mask, i_emb)
        parts = [u_emb, i_emb, hist_rep]
        if u_feats is not None and u_feats.numel() > 0:
            parts.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            parts.append(i_feats)
        dnn_input = torch.cat(parts, dim=-1)
        return self.dnn_features(dnn_input)
