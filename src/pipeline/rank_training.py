from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.rank_dataset import RankDataset
from src.rank.dcn import DCNRanker
from src.rank.din import DINRanker
from src.rank.deepfm import DeepFM
from src.rank.sasrec import SASRecRanker
from src.rank.dcn_din import DCNDINRanker
from src.evaluation import evaluate_ranker_with_candidates


@dataclass
class RankTrainingConfig:
    batch_size: int = 512
    epochs: int = 10
    lr: float = 5e-3
    num_negatives: int = 5
    rank_k: int = 10
    model_type: str = "din"  # choices: dcn, din, deepfm, sasrec, dcn_din
    cross_layers: int = 3
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2
    attention_hidden: tuple = (64, 32)
    sasrec_heads: int = 2
    sasrec_layers: int = 2
    fm_dim: int = 32
    early_stop_patience: Optional[int] = 2


@dataclass
class RankTrainingOutputs:
    ranker: nn.Module
    metrics: dict


def train_ranker_model(
    train_df,
    val_pairs: List[Tuple[int, int]],
    num_items: int,
    user_store,
    item_store,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    user_feat_matrix_cpu: torch.Tensor,
    item_feat_matrix_cpu: torch.Tensor,
    user_candidates,
    test_pairs: List[Tuple[int, int]],
    user_histories: Dict[int, List[int]] | None,
    device: torch.device,
    config: RankTrainingConfig = RankTrainingConfig(),
) -> RankTrainingOutputs:
    user_emb_cpu = user_embeddings.detach().cpu()
    item_emb_cpu = item_embeddings.detach().cpu()
    user_feat_cpu = user_feat_matrix_cpu.detach().cpu()
    item_feat_cpu = item_feat_matrix_cpu.detach().cpu()

    if device.type == "cuda":
        user_emb_train = user_emb_cpu.to(device)
        item_emb_train = item_emb_cpu.to(device)
        user_feat_train = user_feat_cpu.to(device)
        item_feat_train = item_feat_cpu.to(device)
    else:
        user_emb_train = user_emb_cpu
        item_emb_train = item_emb_cpu
        user_feat_train = user_feat_cpu
        item_feat_train = item_feat_cpu

    rank_dataset = RankDataset(
        ratings_df=train_df,
        num_items=num_items,
        user_store=user_store,
        item_store=item_store,
        num_negatives=config.num_negatives,
        user_emb=user_emb_train,
        item_emb=item_emb_train,
        user_feat_tensor=user_feat_train,
        item_feat_tensor=item_feat_train,
        user_candidates=user_candidates,
    )

    rank_loader = DataLoader(
        rank_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type != "cuda",
    )

    user_dim = user_embeddings.shape[1]
    item_dim = item_embeddings.shape[1]
    user_feat_dim = user_feat_cpu.shape[1] if user_feat_cpu.numel() else 0
    item_feat_dim = item_feat_cpu.shape[1] if item_feat_cpu.numel() else 0

    model_type = config.model_type.lower()
    if model_type == "din":
        ranker = DINRanker(
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            attention_hidden=config.attention_hidden,
            dnn_hidden=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    elif model_type == "sasrec":
        ranker = SASRecRanker(
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            max_history=rank_dataset.max_history,
            num_heads=config.sasrec_heads,
            num_layers=config.sasrec_layers,
            dropout=config.dropout,
        ).to(device)
    elif model_type == "deepfm":
        ranker = DeepFM(
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            fm_dim=config.fm_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    elif model_type == "dcn_din":
        ranker = DCNDINRanker(
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            max_history=rank_dataset.max_history,
            cross_layers=config.cross_layers,
            attention_hidden=config.attention_hidden,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    elif model_type == "dcn":
        ranker = DCNRanker(
            user_dim=user_dim,
            item_dim=item_dim,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            cross_layers=config.cross_layers,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown ranking model_type: {config.model_type}")


    optimizer = optim.Adam(ranker.parameters(), lr=config.lr)
    loss_fn = nn.BCELoss()

    best_metrics: Optional[Dict[str, float]] = None
    best_state = None
    patience_counter = 0

    use_history = getattr(ranker, "requires_history", False)

    for epoch in range(config.epochs):
        ranker.train()
        total_loss = 0.0
        for batch in rank_loader:
            optimizer.zero_grad()
            u_emb_batch = batch["u_emb"].to(device)
            i_emb_batch = batch["i_emb"].to(device)
            u_feats_batch = batch["u_feats"].to(device)
            i_feats_batch = batch["i_feats"].to(device)

            if use_history:
                hist_items = batch["hist_items"].to(device)
                hist_mask = hist_items >= 0
                hist_indices = hist_items.clone()
                hist_indices[~hist_mask] = 0
                hist_emb = item_emb_train[hist_indices.long()]
                hist_emb = hist_emb * hist_mask.unsqueeze(-1)
            else:
                hist_emb = None
                hist_mask = None

            preds = ranker(
                u_emb_batch,
                i_emb_batch,
                u_feats=u_feats_batch,
                i_feats=i_feats_batch,
                hist_emb=hist_emb,
                hist_mask=hist_mask,
            )
            loss = loss_fn(preds, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Rank] Epoch {epoch + 1}, Loss = {total_loss / max(len(rank_loader), 1):.4f}")

        ranker.eval()
        with torch.no_grad():
            val_metrics = evaluate_ranker_with_candidates(
                ranker,
                user_emb_cpu,
                item_emb_cpu,
                user_candidates,
                val_pairs,
                rank_k=config.rank_k,
                device=device,
                user_feat_matrix=user_feat_cpu,
                item_feat_matrix=item_feat_cpu,
                user_histories=user_histories,
                max_history=rank_dataset.max_history,
            )

        val_recall = val_metrics.get("recall@k", 0.0)
        print(f"[Rank] Val recall@{config.rank_k} = {val_recall:.4f}")

        if best_metrics is None or val_recall > best_metrics.get("recall@k", 0.0) + 1e-5:
            best_metrics = val_metrics
            best_state = ranker.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if (
                config.early_stop_patience is not None
                and patience_counter >= config.early_stop_patience
            ):
                print(
                    f"[Rank] Early stopping at epoch {epoch + 1}; best val recall@{config.rank_k}="
                    f"{best_metrics.get('recall@k', 0.0):.4f}"
                )
                break

    if best_state is not None:
        ranker.load_state_dict(best_state)
        metrics = best_metrics
    else:
        ranker.eval()
        with torch.no_grad():
            metrics = evaluate_ranker_with_candidates(
                ranker,
                user_emb_cpu,
                item_emb_cpu,
                user_candidates,
                test_pairs,
                rank_k=config.rank_k,
                device=device,
                user_feat_matrix=user_feat_cpu,
                item_feat_matrix=item_feat_cpu,
                user_histories=user_histories,
                max_history=rank_dataset.max_history,
            )

    return RankTrainingOutputs(ranker=ranker, metrics=metrics)
