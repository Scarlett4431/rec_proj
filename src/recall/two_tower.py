import torch
import torch.nn as nn

class Tower(nn.Module):
    """A tower for user or item that combines embedding + extra features."""
    def __init__(self, embed_dim, extra_dim=0, hidden_dim=64):
        super().__init__()
        input_dim = embed_dim + extra_dim
        if extra_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)  # project back to embed_dim
            )
        else:
            self.mlp = None

    def forward(self, emb, extra_feats=None):
        if self.mlp is None:
            return emb
        if extra_feats is not None:
            x = torch.cat([emb, extra_feats], dim=1)
        else:
            x = emb
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64,
                 user_extra_dim=0, item_extra_dim=0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)

        self.user_tower = Tower(embed_dim, user_extra_dim, hidden_dim=128)
        self.item_tower = Tower(embed_dim, item_extra_dim, hidden_dim=128)

    def forward(self, user_ids, item_ids,
                user_feats=None, item_feats=None):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        u_vec = self.user_tower(u, user_feats)
        v_vec = self.item_tower(v, item_feats)
        return u_vec, v_vec

    def user_embed(self, user_ids, user_feats=None):
        u = self.user_embedding(user_ids)
        return self.user_tower(u, user_feats)

    def item_embed(self, item_ids, item_feats=None):
        v = self.item_embedding(item_ids)
        return self.item_tower(v, item_feats)