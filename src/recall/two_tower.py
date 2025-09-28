import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    Two-Tower model for recall:
      - User tower: user ID embedding + side features
      - Item tower: item ID embedding + side features
    """

    def __init__(self, num_users, num_items, embed_dim,
                 user_extra_dim=0, item_extra_dim=0,
                 hidden_dims=None,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dims = hidden_dims or [embed_dim, embed_dim]

        # --- Embedding tables for IDs ---
        self.user_emb = nn.Embedding(num_users + 1, embed_dim)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim)

        # --- Projection layers ---
        user_input_dim = embed_dim + user_extra_dim
        item_input_dim = embed_dim + item_extra_dim
        self.user_mlp = self._build_mlp(user_input_dim, embed_dim, hidden_dims, dropout)
        self.item_mlp = self._build_mlp(item_input_dim, embed_dim, hidden_dims, dropout)

        # Optional: normalization helps training stability
        self.user_norm = nn.LayerNorm(embed_dim)
        self.item_norm = nn.LayerNorm(embed_dim)

    def forward(self, user_ids, item_ids, user_feats=None, item_feats=None):
        """
        Compute embeddings for a batch of (user, item) pairs.
        Args:
            user_ids: [B]
            item_ids: [B]
            user_feats: [B, user_extra_dim] or None
            item_feats: [B, item_extra_dim] or None
        """
        u_emb = self.user_emb(user_ids)  # [B, D]
        i_emb = self.item_emb(item_ids)  # [B, D]

        if user_feats is not None and user_feats.size(1) > 0:
            u_emb = torch.cat([u_emb, user_feats], dim=1)  # [B, D+Fu]
        if item_feats is not None and item_feats.size(1) > 0:
            i_emb = torch.cat([i_emb, item_feats], dim=1)  # [B, D+Fi]

        u_emb = self.user_mlp(u_emb)
        i_emb = self.item_mlp(i_emb)

        u_emb = self.user_norm(u_emb)
        i_emb = self.item_norm(i_emb)

        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb

    def user_embed(self, user_ids, user_feats=None):
        """
        Compute embeddings for users only.
        Args:
            user_ids: [N]
            user_feats: [N, Fu] or None
        """
        u_emb = self.user_emb(user_ids)
        if user_feats is not None and user_feats.size(1) > 0:
            u_emb = torch.cat([u_emb, user_feats], dim=1)
        u_emb = self.user_mlp(u_emb)
        u_emb = self.user_norm(u_emb)
        return F.normalize(u_emb, p=2, dim=1)

    def item_embed(self, item_ids, item_feats=None):
        """
        Compute embeddings for items only.
        Args:
            item_ids: [M]
            item_feats: [M, Fi] or None
        """
        i_emb = self.item_emb(item_ids)
        if item_feats is not None and item_feats.size(1) > 0:
            i_emb = torch.cat([i_emb, item_feats], dim=1)
        i_emb = self.item_mlp(i_emb)
        i_emb = self.item_norm(i_emb)
        return F.normalize(i_emb, p=2, dim=1)

    @staticmethod
    def _build_mlp(input_dim, output_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
