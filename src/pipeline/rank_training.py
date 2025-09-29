from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.rank_dataset import RankDataset
from src.rank.dcn import DCNRanker
from src.evaluation import evaluate_ranker_with_candidates


@dataclass
class RankTrainingConfig:
    batch_size: int = 512
    epochs: int = 3
    lr: float = 1e-3
    num_negatives: int = 5
    rank_k: int = 10
    cross_layers: int = 3
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2


@dataclass
class RankTrainingOutputs:
    ranker: nn.Module
    metrics: dict


def train_ranker_model(
    train_df,
    num_items: int,
    user_store,
    item_store,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    user_feat_matrix_cpu: torch.Tensor,
    item_feat_matrix_cpu: torch.Tensor,
    user_candidates,
    test_pairs: List[Tuple[int, int]],
    device: torch.device,
    config: RankTrainingConfig = RankTrainingConfig(),
) -> RankTrainingOutputs:
    rank_dataset = RankDataset(
        ratings_df=train_df,
        num_items=num_items,
        user_store=user_store,
        item_store=item_store,
        num_negatives=config.num_negatives,
        user_emb=user_embeddings.detach().cpu(),
        item_emb=item_embeddings.detach().cpu(),
        user_feat_tensor=user_feat_matrix_cpu,
        item_feat_tensor=item_feat_matrix_cpu,
        user_candidates=user_candidates,
    )

    rank_loader = DataLoader(
        rank_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    user_dim = user_embeddings.shape[1]
    item_dim = item_embeddings.shape[1]
    user_feat_dim = user_feat_matrix_cpu.shape[1] if user_feat_matrix_cpu.numel() else 0
    item_feat_dim = item_feat_matrix_cpu.shape[1] if item_feat_matrix_cpu.numel() else 0

    ranker = DCNRanker(
        user_dim=user_dim,
        item_dim=item_dim,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        cross_layers=config.cross_layers,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(ranker.parameters(), lr=config.lr)
    loss_fn = nn.BCELoss()

    for epoch in range(config.epochs):
        ranker.train()
        total_loss = 0.0
        for batch in rank_loader:
            optimizer.zero_grad()
            preds = ranker(
                batch["u_emb"].to(device),
                batch["i_emb"].to(device),
                u_feats=batch["u_feats"].to(device),
                i_feats=batch["i_feats"].to(device),
            )
            loss = loss_fn(preds, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Rank] Epoch {epoch + 1}, Loss = {total_loss / max(len(rank_loader), 1):.4f}")

    ranker.eval()
    with torch.no_grad():
        metrics = evaluate_ranker_with_candidates(
            ranker,
            user_embeddings.cpu(),
            item_embeddings.cpu(),
            user_candidates,
            test_pairs,
            rank_k=config.rank_k,
            device=device,
            user_feat_matrix=user_feat_matrix_cpu,
            item_feat_matrix=item_feat_matrix_cpu,
        )

    return RankTrainingOutputs(ranker=ranker, metrics=metrics)
