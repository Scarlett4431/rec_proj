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
        attention_hidden=(64, 32),
        dnn_hidden=(128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.embed_dim = item_dim  # assume provided embeddings already sized appropriately
        self.requires_history = True

        self.attention = AttentionUnit(item_dim, attention_hidden)

        dnn_input_dim = user_dim + item_dim + item_dim + user_feat_dim + item_feat_dim
        layers = []
        prev_dim = dnn_input_dim
        for hidden in dnn_hidden:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        hist_emb: torch.Tensor,
        hist_mask: torch.Tensor,
        u_feats: torch.Tensor = None,
        i_feats: torch.Tensor = None,
    ) -> torch.Tensor:
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
        dnn_input = torch.cat(parts, dim=-1)
        logits = self.dnn(dnn_input)
        return torch.sigmoid(logits).squeeze(-1)
