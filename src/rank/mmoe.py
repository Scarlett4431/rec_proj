import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from src.rank.dcn import DCNRanker
from src.rank.deepfm import DeepFM


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = dim
    return nn.Sequential(*layers) if layers else nn.Identity()


class MMOERanker(nn.Module):
    """Mixture-of-Experts ranker that supports multiple base ranking backbones."""

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
        self.base_model_type = base_model_type.lower()
        self.num_experts = num_experts
        expert_output_dim = expert_hidden[-1] if expert_hidden else self.input_dim

        self.experts = nn.ModuleList(
            build_mlp(self.input_dim, expert_hidden, dropout)
            for _ in range(num_experts)
        )
        self.gate_rank = nn.Linear(self.input_dim, num_experts)
        self.gate_rating = nn.Linear(self.input_dim, num_experts)

        if self.base_model_type == "mlp":
            self.rank_tower = build_mlp(expert_output_dim, tower_hidden, dropout)
            self.rank_output = nn.Linear(
                tower_hidden[-1] if tower_hidden else expert_output_dim, 1
            )
            self.rank_backbone = None
            self.requires_history = False
        elif self.base_model_type == "deepfm":
            self.rank_backbone = DeepFM(
                user_dim=user_dim,
                item_dim=item_dim,
                user_feat_dim=user_feat_dim,
                item_feat_dim=item_feat_dim,
                fm_dim=32,
                hidden_dims=tower_hidden or (128,),
                dropout=dropout,
            )
            self.requires_history = getattr(self.rank_backbone, "requires_history", False)
        elif self.base_model_type == "dcn":
            self.rank_backbone = DCNRanker(
                user_dim=user_dim,
                item_dim=item_dim,
                user_feat_dim=user_feat_dim,
                item_feat_dim=item_feat_dim,
                cross_layers=len(tower_hidden) if tower_hidden else 3,
                hidden_dims=tower_hidden or (128,),
                dropout=dropout,
            )
            self.requires_history = getattr(self.rank_backbone, "requires_history", False)
        else:
            raise ValueError(
                f"Unsupported MMOE base_model_type: {self.base_model_type}."
            )

        self.rating_tower = build_mlp(expert_output_dim, tower_hidden, dropout)
        self.rating_output = nn.Linear(
            tower_hidden[-1] if tower_hidden else expert_output_dim, 1
        )

    def forward(
        self,
        user_emb,
        item_emb,
        u_feats=None,
        i_feats=None,
        hist_emb=None,
        hist_mask=None,
    ):
        inputs = [user_emb, item_emb]
        if u_feats is not None:
            inputs.append(u_feats)
        if i_feats is not None:
            inputs.append(i_feats)
        features = torch.cat(inputs, dim=-1)

        expert_outputs = [expert(features) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        rank_weights = torch.softmax(self.gate_rank(features), dim=1).unsqueeze(-1)
        rating_weights = torch.softmax(self.gate_rating(features), dim=1).unsqueeze(-1)

        # Ranking branch
        if self.base_model_type == "mlp":
            rank_input = torch.sum(rank_weights * expert_stack, dim=1)
            rank_hidden = self.rank_tower(rank_input)
            rank_score = self.rank_output(rank_hidden).squeeze(-1)
        else:
            if self.base_model_type == "deepfm":
                rank_score = self.rank_backbone(
                    user_emb, item_emb, u_feats=u_feats, i_feats=i_feats
                )
            elif self.base_model_type == "dcn":
                rank_score = self.rank_backbone(
                    user_emb, item_emb, u_feats=u_feats, i_feats=i_feats
                )
            if isinstance(rank_score, tuple):
                rank_score = rank_score[0]
            rank_score = rank_score.squeeze(-1)

        # Rating branch uses shared experts
        rating_input = torch.sum(rating_weights * expert_stack, dim=1)
        rating_hidden = self.rating_tower(rating_input)
        rating_score = torch.sigmoid(self.rating_output(rating_hidden)).squeeze(-1) * 5.0

        return rank_score, rating_score
