import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """
    Encodes numeric + categorical + bucketized features into a dense vector.
    - numeric: linear projection
    - categorical/bucket: embeddings summed or concatenated
    """

    def __init__(self, numeric_dim, cat_dims=None, bucket_dims=None, embed_dim=16, proj_dim=32):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.cat_dims = cat_dims or {}
        self.bucket_dims = bucket_dims or {}

        # Numeric projection
        self.num_proj = nn.Linear(numeric_dim, proj_dim) if numeric_dim > 0 else None

        # Categorical embeddings
        self.cat_embeds = nn.ModuleDict({
            col: nn.Embedding(size, embed_dim)
            for col, size in self.cat_dims.items()
        })

        # Bucket embeddings
        self.bucket_embeds = nn.ModuleDict({
            col: nn.Embedding(size, embed_dim)
            for col, size in self.bucket_dims.items()
        })

        # Output dimension
        self.out_dim = (proj_dim if self.num_proj else 0) \
                       + len(self.cat_embeds) * embed_dim \
                       + len(self.bucket_embeds) * embed_dim

    def forward(self, feats):
        """
        feats: dict with {"numeric": Tensor[F], "categorical": {col: idx}, "bucket": {col: idx}}
        """
        outs = []

        # Numeric
        if self.num_proj and feats["numeric"].numel() > 0:
            outs.append(self.num_proj(feats["numeric"]))

        # Categorical
        for col, emb in self.cat_embeds.items():
            outs.append(emb(feats["categorical"][col]))

        # Bucketized
        for col, emb in self.bucket_embeds.items():
            outs.append(emb(feats["bucket"][col]))

        return torch.cat(outs, dim=-1) if outs else torch.zeros(0, device=feats["numeric"].device)