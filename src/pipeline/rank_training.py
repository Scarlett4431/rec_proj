import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.rank_dataset import RankDataset, RankDatasetConfig
from src.evaluation import evaluate_ranker_with_candidates
from src.losses import bpr_loss
from src.rank.dcn import DCNRanker
from src.rank.dcn_din import DCNDINRanker
from src.rank.deepfm import DeepFM
from src.rank.din import DINRanker
from src.rank.sasrec import SASRecRanker


@dataclass
class RankTrainingConfig:
    batch_size: int = 512
    epochs: int = 30
    lr: float = 5e-4
    num_negatives: int = 5
    candidate_neg_ratio: float = 0.6
    history_max: int = 50
    rank_k: int = 10
    model_type: str = "deepfm"  # choices: dcn, din, deepfm, sasrec, dcn_din
    cross_layers: int = 3
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2
    attention_hidden: tuple = (64, 32)
    sasrec_heads: int = 2
    sasrec_layers: int = 2
    fm_dim: int = 32
    early_stop_patience: Optional[int] = 3
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    eval_batch_size: int = 4096
    bpr_margin: float = 0.4
    margin_weight: float = 0.1


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
        user_emb_device = user_emb_cpu.to(device, non_blocking=True)
        item_emb_device = item_emb_cpu.to(device, non_blocking=True)
        user_feat_device = user_feat_cpu.to(device, non_blocking=True)
        item_feat_device = item_feat_cpu.to(device, non_blocking=True)
    else:
        user_emb_device = user_emb_cpu
        item_emb_device = item_emb_cpu
        user_feat_device = user_feat_cpu
        item_feat_device = item_feat_cpu

    dataset_cfg = RankDatasetConfig(
        num_negatives=config.num_negatives,
        max_history=config.history_max,
        candidate_neg_ratio=config.candidate_neg_ratio,
    )
    rank_dataset = RankDataset(
        ratings_df=train_df,
        num_items=num_items,
        user_store=user_store,
        item_store=item_store,
        config=dataset_cfg,
        user_candidates=user_candidates,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
        num_workers=config.num_workers,
    )

    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    rank_loader = DataLoader(
        rank_dataset,
        **loader_kwargs,
    )

    if user_feat_device is not None and user_feat_device.numel() == 0:
        user_feat_device = None
    if item_feat_device is not None and item_feat_device.numel() == 0:
        item_feat_device = None
    max_hist = rank_dataset.max_history

    def gather_features(matrix: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
        if matrix is None:
            return None
        return matrix[indices]

    def score_pairs(
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        hist_emb: torch.Tensor | None = None,
        hist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_vecs = user_emb_device[user_idx]
        item_vecs = item_emb_device[item_idx]
        user_feats = gather_features(user_feat_device, user_idx)
        item_feats = gather_features(item_feat_device, item_idx)

        if use_history:
            scores = ranker(
                user_vecs,
                item_vecs,
                u_feats=user_feats,
                i_feats=item_feats,
                hist_emb=hist_emb,
                hist_mask=hist_mask,
            )
        else:
            scores = ranker(
                user_vecs,
                item_vecs,
                u_feats=user_feats,
                i_feats=item_feats,
            )
        return scores.view(-1)

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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_metrics: Optional[Dict[str, float]] = None
    best_state = None
    patience_counter = 0

    use_history = getattr(ranker, "requires_history", False)

    if user_feat_device is not None and user_feat_device.numel() == 0:
        user_feat_device = None
    if item_feat_device is not None and item_feat_device.numel() == 0:
        item_feat_device = None

    max_hist = rank_dataset.max_history

    for epoch in range(config.epochs):
        ranker.train()
        total_loss = 0.0
        history_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        pos_mean_acc = 0.0
        neg_mean_acc = 0.0
        gap_acc = 0.0
        hit_acc = 0.0
        batch_count = 0
        epoch_start = time.time()
        for batch in rank_loader:
            optimizer.zero_grad()
            user_ids = batch["user_id"].to(device, non_blocking=True)
            pos_items = batch["pos_item"].to(device, non_blocking=True)
            neg_items = batch["neg_items"].to(device, non_blocking=True)

            if neg_items.numel() == 0 or neg_items.shape[1] == 0:
                continue

            num_neg = neg_items.shape[1]

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
            pos_scores = score_pairs(user_ids, pos_items, hist_emb, hist_mask)

            neg_flat = neg_items.view(-1)
            neg_user_ids = user_ids.repeat_interleave(num_neg)
            if use_history and hist_emb is not None and hist_mask is not None:
                neg_hist_emb = hist_emb.repeat_interleave(num_neg, dim=0)
                neg_hist_mask = hist_mask.repeat_interleave(num_neg, dim=0)
            else:
                neg_hist_emb = None
                neg_hist_mask = None
            neg_scores = score_pairs(
                neg_user_ids,
                neg_flat,
                neg_hist_emb,
                neg_hist_mask,
            ).view(user_ids.size(0), num_neg)
            forward_time += time.time() - t_fwd

            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()
            gap = pos_mean - neg_mean
            hit_ratio = torch.mean((pos_scores.unsqueeze(-1) > neg_scores).float()).item()
            pos_mean_acc += pos_mean
            neg_mean_acc += neg_mean
            gap_acc += gap
            hit_acc += hit_ratio
            batch_count += 1

            bpr = bpr_loss(pos_scores.unsqueeze(-1), neg_scores)
            pos_mean = pos_scores.mean()
            neg_mean = neg_scores.mean()
            margin_penalty = torch.relu(
                config.bpr_margin - (pos_mean - neg_mean)
            )
            loss = bpr + config.margin_weight * margin_penalty
            t_bwd = time.time()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            backward_time += time.time() - t_bwd
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(batch_count, 1)
        print(f"[Rank] Epoch {epoch + 1}, Loss = {avg_loss:.4f}")
        print(
            f"[Timing][Rank] Epoch {epoch + 1} took {epoch_time:.2f}s "
            f"(history {history_time:.2f}s | forward {forward_time:.2f}s | backward {backward_time:.2f}s)"
        )
        if batch_count > 0:
            print(
                f"[RankDebug] epoch={epoch + 1} pos={pos_mean_acc / batch_count:.4f} "
                f"neg={neg_mean_acc / batch_count:.4f} gap={gap_acc / batch_count:.4f} "
                f"hit>{hit_acc / batch_count:.3f}"
            )

        scheduler.step()

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
            or val_ndcg > best_metrics.get("ndcg@k", 0.0) + 1e-5
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
                    f"[Rank] Early stopping at epoch {epoch + 1}; best val ndcg@{config.rank_k}="
                    f"{best_metrics.get('ndcg@k', 0.0):.4f}"
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
