import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.recall_dataset import RecallDataset
from src.features.feature_utils import encode_cached_batch, encode_features
from src.losses import InBatchSoftmaxLoss
from src.evaluation import evaluate_filtered_faiss
from src.recall.two_tower import TwoTowerModel


@dataclass
class RecallTrainingConfig:
    batch_size: int = 1024
    warmup_epochs: int = 8
    easy_neg_samples: int = 3
    embed_dim: int = 64
    tower_dropout: float = 0.1
    emb_lr: float = 5e-3
    tower_lr: float = 5e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1


@dataclass
class RecallTrainingOutputs:
    model: TwoTowerModel
    user_encoder: nn.Module
    item_encoder: nn.Module
    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor


def train_two_tower_model(
    train_df,
    num_users: int,
    num_items: int,
    user_consumed: Dict[int, set],
    val_users: List[int],
    val_pairs: List[Tuple[int, int]],
    feature_components,
    device: torch.device,
    config: RecallTrainingConfig = RecallTrainingConfig(),
) -> RecallTrainingOutputs:
    recall_dataset = RecallDataset(
        train_df,
        feature_components.user_store,
        feature_components.item_store,
        easy_neg_samples=config.easy_neg_samples,
        num_items=num_items,
    )
    recall_loader = DataLoader(
        recall_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    projected_title_dim = 0
    if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
        projected_title_dim = feature_components.item_title_proj.out_features

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=config.embed_dim,
        user_extra_dim=feature_components.user_encoder.out_dim,
        item_extra_dim=feature_components.item_encoder.out_dim + projected_title_dim,
        dropout=config.tower_dropout,
    ).to(device)

    embedding_params = list(model.user_emb.parameters()) + list(model.item_emb.parameters())
    tower_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("user_emb") and not name.startswith("item_emb")
    ]
    encoder_params = (
        list(feature_components.user_encoder.parameters())
        + list(feature_components.item_encoder.parameters())
    )
    if feature_components.item_title_proj is not None:
        encoder_params += list(feature_components.item_title_proj.parameters())

    optimizer = optim.Adam(
        [
            {"params": embedding_params, "lr": config.emb_lr, "weight_decay": 0.0},
            {
                "params": tower_params + encoder_params,
                "lr": config.tower_lr,
                "weight_decay": config.weight_decay,
            },
        ]
    )
    loss_fn = InBatchSoftmaxLoss(temperature=config.temperature)

    def export_embeddings():
        model.eval()
        feature_components.user_encoder.eval()
        feature_components.item_encoder.eval()
        with torch.no_grad():
            user_ids = torch.arange(num_users, dtype=torch.long)
            item_ids = torch.arange(num_items, dtype=torch.long)

            user_feats_raw = feature_components.user_store.get_batch(
                user_ids.tolist(),
                max_multi_lengths=feature_components.user_multi_max,
            )
            item_feats_raw = feature_components.item_store.get_batch(
                item_ids.tolist(),
                max_multi_lengths=feature_components.item_multi_max,
            )

            user_side = encode_features(feature_components.user_encoder, user_feats_raw, device)
            item_side = encode_features(feature_components.item_encoder, item_feats_raw, device)
            if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
                projected_titles = feature_components.item_title_proj(feature_components.item_title_embeddings)
                item_side = torch.cat([item_side, projected_titles], dim=1)

            user_emb = model.user_embed(user_ids.to(device), user_feats=user_side)
            item_emb = model.item_embed(item_ids.to(device), item_feats=item_side)
        model.train()
        feature_components.user_encoder.train()
        feature_components.item_encoder.train()
        return user_emb, item_emb

    latest_user_emb = None
    latest_item_emb = None

    for epoch in range(config.warmup_epochs):
        model.train()
        total_loss = 0.0

        for batch in recall_loader:
            user_ids = batch["user_id"].to(device)
            pos_items = batch["pos_item"].to(device)

            user_feats = encode_cached_batch(
                feature_components.user_feature_cache,
                feature_components.user_encoder,
                user_ids,
                device,
            )
            item_feats = encode_cached_batch(
                feature_components.item_feature_cache,
                feature_components.item_encoder,
                pos_items,
                device,
            )
            if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
                projected_titles = feature_components.item_title_proj(
                    feature_components.item_title_embeddings[pos_items]
                )
                item_feats = torch.cat([item_feats, projected_titles], dim=1)

            optimizer.zero_grad()
            user_out, pos_out = model(
                user_ids,
                pos_items,
                user_feats=user_feats,
                item_feats=item_feats,
            )

            neg_items = batch["easy_neg_items"].to(device)
            if neg_items.numel() > 0:
                timer_start = time.perf_counter()
                valid_mask = neg_items >= 0
                if valid_mask.any():
                    neg_item_ids = neg_items[valid_mask]
                    neg_feats = encode_cached_batch(
                        feature_components.item_feature_cache,
                        feature_components.item_encoder,
                        neg_item_ids,
                        device,
                    )
                    if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
                        projected_titles = feature_components.item_title_proj(
                            feature_components.item_title_embeddings[neg_item_ids]
                        )
                        neg_feats = torch.cat([neg_feats, projected_titles], dim=1)
                    neg_emb = model.item_embed(neg_item_ids, item_feats=neg_feats)
                    pos_out = torch.cat([pos_out, neg_emb], dim=0)
                _ = time.perf_counter() - timer_start

            loss = loss_fn(user_out, pos_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        latest_user_emb, latest_item_emb = export_embeddings()
        val_metrics = evaluate_filtered_faiss(
            latest_user_emb,
            latest_item_emb,
            val_users,
            val_pairs,
            user_consumed,
            num_items,
            k=100,
        )
        print(
            f"[Recall] Warm-up Epoch {epoch + 1}/{config.warmup_epochs}, Loss = {total_loss / max(len(recall_loader), 1):.4f}, "
            f"val_recall@100={val_metrics.get('recall@k', 0.0):.4f}, val_ndcg@100={val_metrics.get('ndcg@k', 0.0):.4f}"
        )

    user_emb, item_emb = latest_user_emb, latest_item_emb
    if user_emb is None or item_emb is None:
        user_emb, item_emb = export_embeddings()

    return RecallTrainingOutputs(
        model=model,
        user_encoder=feature_components.user_encoder,
        item_encoder=feature_components.item_encoder,
        user_embeddings=user_emb,
        item_embeddings=item_emb,
    )
