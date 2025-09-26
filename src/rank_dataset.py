import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from src.features.feature_store import FeatureStore

class RankDataset(Dataset):
    def __init__(self, ratings_df, num_items, user_features, item_features,
                 num_negatives=5, user_emb=None, item_emb=None):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_emb = user_emb
        self.item_emb = item_emb

        # feature stores
        self.user_store = FeatureStore(user_features, "user_idx")
        self.item_store = FeatureStore(item_features, "item_idx")
        self.user_feat_dim = self.user_store.dim
        self.item_feat_dim = self.item_store.dim
        self.extra_dim = self.user_feat_dim + self.item_feat_dim

        # interactions
        self.user_item_pairs = ratings_df[["user_idx", "item_idx"]].values
        self.user_rated = defaultdict(set)
        for u, i in self.user_item_pairs:
            self.user_rated[u].add(i)

        # pos + neg samples
        self.samples = []
        for u, i in self.user_item_pairs:
            self.samples.append((u, i, 1))
            for _ in range(num_negatives):
                j = np.random.randint(1, num_items + 1)
                while j in self.user_rated[u]:
                    j = np.random.randint(1, num_items + 1)
                self.samples.append((u, j, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, label = self.samples[idx]

        u_emb = self.user_emb[u]
        i_emb = self.item_emb[i]

        u_feats = self.user_store.get(u)
        i_feats = self.item_store.get(i)
        extra_feats = torch.cat([u_feats, i_feats], dim=0).clone() if self.extra_dim > 0 else torch.zeros(0)

        return (
            u_emb.float(),
            i_emb.float(),
            extra_feats,
            torch.tensor(label, dtype=torch.float32)
        )