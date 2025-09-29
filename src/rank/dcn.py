import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0, xl: [B, D]
        cross = torch.matmul(xl, self.weight)  # [B]
        return x0 * cross.unsqueeze(-1) + self.bias + xl


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DCNRanker(nn.Module):
    """Deep & Cross Network for ranking with dense inputs."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        cross_layers: int = 3,
        hidden_dims = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim
        self.input_dim = input_dim

        self.cross_net = CrossNetwork(input_dim, cross_layers)

        deep_layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden))
            deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.deep_net = nn.Sequential(*deep_layers) if deep_layers else nn.Identity()

        combined_dim = input_dim + (hidden_dims[-1] if hidden_dims else input_dim)
        self.output_layer = nn.Linear(combined_dim, 1)

    def _concat_features(self, u_emb, i_emb, u_feats=None, i_feats=None):
        parts = [u_emb, i_emb]
        if u_feats is not None and u_feats.numel() > 0:
            parts.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            parts.append(i_feats)
        return torch.cat(parts, dim=-1)

    def forward(self, u_emb, i_emb, u_feats=None, i_feats=None):
        x = self._concat_features(u_emb, i_emb, u_feats, i_feats)
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.output_layer(combined)
        return torch.sigmoid(logits).squeeze(-1)
