import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Encodes heterogeneous feature groups into a dense vector."""

    def __init__(self, numeric_dim,
                 cat_dims=None,
                 bucket_dims=None,
                 multi_cat_dims=None,
                 embed_dim=16,
                 bucket_embed_dim=None,
                 multi_embed_dim=None,
                 proj_dim=32,
                 pooling="mean"):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.cat_dims = cat_dims or {}
        self.bucket_dims = bucket_dims or {}
        self.multi_cat_dims = multi_cat_dims or {}
        self.pooling = pooling

        bucket_embed_dim = bucket_embed_dim or embed_dim
        multi_embed_dim = multi_embed_dim or embed_dim

        self.num_proj = nn.Linear(numeric_dim, proj_dim) if numeric_dim > 0 else None

        self.cat_embeds = nn.ModuleDict({
            col: nn.Embedding(size, embed_dim, padding_idx=0)
            for col, size in self.cat_dims.items()
        })

        self.bucket_embeds = nn.ModuleDict({
            col: nn.Embedding(size, bucket_embed_dim, padding_idx=0)
            for col, size in self.bucket_dims.items()
        })

        self.multi_cat_embeds = nn.ModuleDict({
            col: nn.Embedding(size, multi_embed_dim, padding_idx=0)
            for col, size in self.multi_cat_dims.items()
        })

        self.out_dim = (proj_dim if self.num_proj else 0) \
                       + len(self.cat_embeds) * embed_dim \
                       + len(self.bucket_embeds) * bucket_embed_dim \
                       + len(self.multi_cat_embeds) * multi_embed_dim

    def forward(self, feats):
        outs = []
        device = None
        batch_size = None

        numeric = feats.get("numeric")
        if numeric is not None and numeric.numel() > 0:
            if numeric.dim() == 1:
                numeric = numeric.unsqueeze(0)
            device = numeric.device
            batch_size = numeric.size(0)
            if self.num_proj:
                outs.append(self.num_proj(numeric))
        elif self.num_proj:
            pass

        categorical = feats.get("categorical", {})
        for col, emb in self.cat_embeds.items():
            col_tensor = categorical.get(col)
            if col_tensor is None:
                continue
            if col_tensor.dim() == 0:
                col_tensor = col_tensor.unsqueeze(0)
            if batch_size is None:
                batch_size = col_tensor.size(0)
            device = col_tensor.device
            outs.append(emb(col_tensor))

        bucket = feats.get("bucket", {})
        for col, emb in self.bucket_embeds.items():
            col_tensor = bucket.get(col)
            if col_tensor is None:
                continue
            if col_tensor.dim() == 0:
                col_tensor = col_tensor.unsqueeze(0)
            if batch_size is None:
                batch_size = col_tensor.size(0)
            device = col_tensor.device
            outs.append(emb(col_tensor))

        multi = feats.get("multi_categorical", {})
        multi_lengths = feats.get("multi_lengths", {})
        for col, emb in self.multi_cat_embeds.items():
            col_tensor = multi.get(col)
            if col_tensor is None or col_tensor.numel() == 0:
                continue
            if col_tensor.dim() == 1:
                col_tensor = col_tensor.unsqueeze(0)
            if batch_size is None:
                batch_size = col_tensor.size(0)
            device = col_tensor.device

            emb_vals = emb(col_tensor)  # [B, L, D]
            mask = (col_tensor != 0).unsqueeze(-1)
            if self.pooling == "sum":
                pooled = (emb_vals * mask).sum(dim=1)
            else:
                summed = (emb_vals * mask).sum(dim=1)
                lengths = multi_lengths.get(col)
                if lengths is None:
                    lengths = mask.sum(dim=1)
                else:
                    if lengths.dim() == 1:
                        lengths = lengths.unsqueeze(-1)
                pooled = summed / lengths.clamp(min=1e-6)
            outs.append(pooled)

        if not outs:
            if batch_size is None:
                return torch.zeros((0, 0), device=device or torch.device("cpu"))
            return torch.zeros((batch_size, 0), device=device or torch.device("cpu"))

        return torch.cat(outs, dim=-1)
