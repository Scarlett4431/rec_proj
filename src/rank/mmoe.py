import torch
import torch.nn as nn
from typing import Sequence

from src.rank.deepfm import DeepFM
from src.rank.dcn import DCNRanker
from src.rank.rank_mlp import RankerMLP


def _concat_inputs(user_emb, item_emb, u_feats=None, i_feats=None):
    parts = [user_emb, item_emb]
    if u_feats is not None and u_feats.numel() > 0:
        parts.append(u_feats)
    if i_feats is not None and i_feats.numel() > 0:
        parts.append(i_feats)
    return torch.cat(parts, dim=-1)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = dim
    return nn.Sequential(*layers) if layers else nn.Identity()


class _ExpertAdapter(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        if not hasattr(base_model, "extract_features"):
            raise AttributeError("Base model must implement extract_features.")
        if not hasattr(base_model, "feature_dim"):
            raise AttributeError("Base model must expose feature_dim.")
        self.output_dim = base_model.feature_dim

    def forward(self, user_emb, item_emb, u_feats=None, i_feats=None, hist_emb=None, hist_mask=None):
        return self.base_model.extract_features(user_emb, item_emb, u_feats, i_feats)


def _create_experts(
    base_model: str,
    num_experts: int,
    user_dim: int,
    item_dim: int,
    user_feat_dim: int,
    item_feat_dim: int,
    expert_hidden: Sequence[int],
    dropout: float,
) -> nn.ModuleList:
    base = base_model.lower()
    experts = []
    for _ in range(num_experts):
        if base == "mlp":
            model = RankerMLP(
                user_dim=user_dim,
                item_dim=item_dim,
                user_feat_dim=user_feat_dim,
                item_feat_dim=item_feat_dim,
                hidden_dims=expert_hidden,
                dropout=dropout,
            )
        elif base == "deepfm":
            fm_dim = max(16, expert_hidden[-1] if expert_hidden else 16)
            model = DeepFM(
                user_dim=user_dim,
                item_dim=item_dim,
                user_feat_dim=user_feat_dim,
                item_feat_dim=item_feat_dim,
                fm_dim=fm_dim,
                hidden_dims=expert_hidden,
                dropout=dropout,
            )
        elif base == "dcn":
            model = DCNRanker(
                user_dim=user_dim,
                item_dim=item_dim,
                user_feat_dim=user_feat_dim,
                item_feat_dim=item_feat_dim,
                cross_layers=3,
                hidden_dims=expert_hidden,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported MMOE base_model_type: {base_model}")
        experts.append(_ExpertAdapter(model))
    return nn.ModuleList(experts)


class MMOERanker(nn.Module):
    """Mixture-of-Experts ranker emitting ranking and rating scores."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        num_experts: int = 4,
        expert_hidden: Sequence[int] = (256, 128),
        tower_hidden: Sequence[int] = (128,),
        dropout: float = 0.2,
        base_model_type: str = "mlp",
    ):
        super().__init__()

        self.input_dim = user_dim + item_dim + user_feat_dim + item_feat_dim
        self.experts = _create_experts(
            base_model=base_model_type,
            num_experts=num_experts,
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            expert_hidden=expert_hidden,
            dropout=dropout,
        )
        self.expert_output_dim = self.experts[0].output_dim

        self.gate_rank = nn.Linear(self.input_dim, num_experts)
        self.gate_rating = nn.Linear(self.input_dim, num_experts)

        self.rank_tower = _build_mlp(self.expert_output_dim, tower_hidden, dropout)
        self.rank_output = nn.Linear(
            tower_hidden[-1] if tower_hidden else self.expert_output_dim, 1
        )

        self.rating_tower = _build_mlp(self.expert_output_dim, tower_hidden, dropout)
        self.rating_output = nn.Linear(
            tower_hidden[-1] if tower_hidden else self.expert_output_dim, 1
        )

        self.requires_history = False

    def forward(
        self,
        user_emb,
        item_emb,
        u_feats=None,
        i_feats=None,
        hist_emb=None,
        hist_mask=None,
    ):
        features = _concat_inputs(user_emb, item_emb, u_feats, i_feats)

        expert_outputs = torch.stack(
            [
                expert(user_emb, item_emb, u_feats, i_feats, hist_emb, hist_mask)
                for expert in self.experts
            ],
            dim=1,
        )

        rank_weights = torch.softmax(self.gate_rank(features), dim=1).unsqueeze(-1)
        rating_weights = torch.softmax(self.gate_rating(features), dim=1).unsqueeze(-1)

        rank_input = torch.sum(rank_weights * expert_outputs, dim=1)
        rank_hidden = self.rank_tower(rank_input)
        rank_score = self.rank_output(rank_hidden).squeeze(-1)

        rating_input = torch.sum(rating_weights * expert_outputs, dim=1)
        rating_hidden = self.rating_tower(rating_input)
        rating_score = torch.sigmoid(self.rating_output(rating_hidden)).squeeze(-1) * 5.0

        return rank_score, rating_score
