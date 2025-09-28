import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for recall:
      - User tower: user ID embedding + side features
      - Item tower: item ID embedding + side features
    """

    def __init__(self, num_users, num_items, embed_dim,
                 user_extra_dim=0, item_extra_dim=0):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Embedding tables for IDs ---
        self.user_emb = nn.Embedding(num_users + 1, embed_dim)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim)

        # --- Projection layers ---
        self.user_proj = nn.Linear(embed_dim + user_extra_dim, embed_dim)
        self.item_proj = nn.Linear(embed_dim + item_extra_dim, embed_dim)

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

        u_emb = self.user_proj(u_emb)  # [B, D]
        i_emb = self.item_proj(i_emb)  # [B, D]

        u_emb = self.user_norm(u_emb)
        i_emb = self.item_norm(i_emb)

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
        u_emb = self.user_proj(u_emb)
        return self.user_norm(u_emb)

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
        i_emb = self.item_proj(i_emb)
        return self.item_norm(i_emb)