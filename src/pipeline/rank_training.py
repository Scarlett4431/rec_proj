import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.rank_dataset import RankDataset
from src.evaluation import evaluate_ranker_with_candidates
from src.rank.dcn import DCNRanker
from src.rank.dcn_din import DCNDINRanker
from src.rank.deepfm import DeepFM
from src.rank.din import DINRanker
from src.rank.sasrec import SASRecRanker


@dataclass
class RankTrainingConfig:
    batch_size: int = 512
    epochs: int = 10
    lr: float = 5e-3
    num_negatives: int = 5
    rank_k: int = 10
    model_type: str = "dcn"  # choices: dcn, din, deepfm, sasrec, dcn_din
    cross_layers: int = 3
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2
    attention_hidden: tuple = (64, 32)
    sasrec_heads: int = 2
    sasrec_layers: int = 2
    fm_dim: int = 32
    early_stop_patience: Optional[int] = 2
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    eval_batch_size: int = 4096


@dataclass
class RankTrainingOutputs:
    ranker: nn.Module
    metrics: dict
    max_history: int


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
    skip_missing_eval: bool = False,
) -> RankTrainingOutputs:
    user_emb_cpu = user_embeddings.detach().contiguous().cpu()
    item_emb_cpu = item_embeddings.detach().contiguous().cpu()
    user_feat_cpu = user_feat_matrix_cpu.detach().contiguous().cpu()
    item_feat_cpu = item_feat_matrix_cpu.detach().contiguous().cpu()

    if device.type == "cuda":
        user_emb_dataset = user_emb_cpu.pin_memory()
        item_emb_dataset = item_emb_cpu.pin_memory()
        user_feat_dataset = user_feat_cpu.pin_memory()
        item_feat_dataset = item_feat_cpu.pin_memory()

        user_emb_device = user_emb_cpu.to(device, non_blocking=True)
        item_emb_device = item_emb_cpu.to(device, non_blocking=True)
        user_feat_device = user_feat_cpu.to(device, non_blocking=True)
        item_feat_device = item_feat_cpu.to(device, non_blocking=True)
    else:
        user_emb_dataset = user_emb_cpu
        item_emb_dataset = item_emb_cpu
        user_feat_dataset = user_feat_cpu
        item_feat_dataset = item_feat_cpu

        user_emb_device = user_emb_cpu
        item_emb_device = item_emb_cpu
        user_feat_device = user_feat_cpu
        item_feat_device = item_feat_cpu

    rank_dataset = RankDataset(
        ratings_df=train_df,
        num_items=num_items,
        user_store=user_store,
        item_store=item_store,
        num_negatives=config.num_negatives,
        user_emb=user_emb_dataset,
        item_emb=item_emb_dataset,
        user_feat_tensor=user_feat_dataset,
        item_feat_tensor=item_feat_dataset,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
        num_workers=config.num_workers,
    )

    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    rank_loader = DataLoader(
        rank_dataset,
        **loader_kwargs,
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
        history_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        epoch_start = time.time()
        for batch in rank_loader:
            optimizer.zero_grad()
            u_emb_batch = batch["u_emb"].to(device, non_blocking=True)
            i_emb_batch = batch["i_emb"].to(device, non_blocking=True)
            u_feats_batch = batch["u_feats"].to(device, non_blocking=True)
            i_feats_batch = batch["i_feats"].to(device, non_blocking=True)

            if use_history:
                t_hist = time.time()
                hist_items = batch["hist_items"].to(device, non_blocking=True)
                hist_mask = hist_items >= 0
                hist_indices = hist_items.clone()
                hist_indices[~hist_mask] = 0
                hist_emb = item_emb_device[hist_indices.long()]
                hist_emb = hist_emb * hist_mask.unsqueeze(-1)
                history_time += time.time() - t_hist
            else:
                hist_emb = None
                hist_mask = None

            t_fwd = time.time()
            if use_history:
                preds = ranker(
                    u_emb_batch,
                    i_emb_batch,
                    u_feats=u_feats_batch,
                    i_feats=i_feats_batch,
                    hist_emb=hist_emb,
                    hist_mask=hist_mask,
                )
            else:
                preds = ranker(
                    u_emb_batch,
                    i_emb_batch,
                    u_feats=u_feats_batch,
                    i_feats=i_feats_batch,
                )
            forward_time += time.time() - t_fwd
            loss = loss_fn(preds, batch["label"].to(device, non_blocking=True))
            t_bwd = time.time()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            backward_time += time.time() - t_bwd
        epoch_time = time.time() - epoch_start
        print(
            f"[Rank] Epoch {epoch + 1}, Loss = {total_loss / max(len(rank_loader), 1):.4f}"
        )
        print(
            f"[Timing][Rank] Epoch {epoch + 1} took {epoch_time:.2f}s "
            f"(history {history_time:.2f}s | forward {forward_time:.2f}s | backward {backward_time:.2f}s)"
        )

        ranker.eval()
        with torch.no_grad():
            eval_start = time.time()
            val_metrics = evaluate_ranker_with_candidates(
                ranker,
                user_emb_device,
                item_emb_device,
                user_candidates,
                val_pairs,
                rank_k=config.rank_k,
                device=device,
                user_feat_matrix=user_feat_device,
                item_feat_matrix=item_feat_device,
                user_histories=user_histories,
                max_history=rank_dataset.max_history,
                batch_size=config.eval_batch_size,
                skip_missing_candidates=skip_missing_eval,
            )
            eval_time = time.time() - eval_start

        val_recall = val_metrics.get("recall@k", 0.0)
        val_ndcg = val_metrics.get("ndcg@k", 0.0)
        val_gauc = val_metrics.get("gauc@k", 0.0)
        print(
            f"[Rank] Val metrics @ {config.rank_k}: "
            f"recall={val_recall:.4f}, ndcg={val_ndcg:.4f}, gauc={val_gauc:.4f}"
        )
        print(f"[Timing][Rank] Validation took {eval_time:.2f}s")

        if (
            best_metrics is None
            or val_recall > best_metrics.get("recall@k", 0.0) + 1e-5
        ):
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
            eval_start = time.time()
            metrics = evaluate_ranker_with_candidates(
                ranker,
                user_emb_device,
                item_emb_device,
                user_candidates,
                test_pairs,
                rank_k=config.rank_k,
                device=device,
                user_feat_matrix=user_feat_device,
                item_feat_matrix=item_feat_device,
                user_histories=user_histories,
                max_history=rank_dataset.max_history,
                batch_size=config.eval_batch_size,
                skip_missing_candidates=skip_missing_eval,
            )
            print(
                f"[Timing][Rank] Test evaluation took {time.time() - eval_start:.2f}s"
            )

    return RankTrainingOutputs(
        ranker=ranker,
        metrics=metrics,
        max_history=rank_dataset.max_history,
    )
