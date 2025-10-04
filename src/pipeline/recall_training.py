import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    batch_size: int = 256
    k: int = 100
    warmup_epochs: int = 20
    easy_neg_samples: int = 3
    tail_neg_samples: int = 3
    hard_neg_samples: int = 0        
    hard_neg_k: int = 50             
    tail_sampling_alpha: float = 1
    tail_sampling_smoothing: float = 1.0
    hard_neg_epochs: int = 30
    embed_dim: int = 64
    tower_dropout: float = 0.1
    emb_lr: float = 5e-3
    tower_lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.2
    early_stop_patience: Optional[int] = 3
    debug_similarity_samples: int = 128


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
    warmup_dataset = RecallDataset(
        train_df,
        feature_components.user_store,
        feature_components.item_store,
        easy_neg_samples=config.easy_neg_samples,
        hard_neg_k=0,
        hard_neg_samples=0,
        tail_neg_samples=config.tail_neg_samples,
        tail_sampling_alpha=config.tail_sampling_alpha,
        tail_sampling_smoothing=config.tail_sampling_smoothing,
        num_items=num_items,
    )
    warmup_loader = DataLoader(
        warmup_dataset,
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

    k = config.k

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
            print(f"User side shape: {user_side.shape}")
            item_side = encode_features(feature_components.item_encoder, item_feats_raw, device)
            if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
                projected_titles = feature_components.item_title_proj(feature_components.item_title_embeddings)
                item_side = torch.cat([item_side, projected_titles], dim=1)
                print(f"Item side shape with titles: {item_side.shape}")

            user_emb = model.user_embed(user_ids.to(device), user_feats=user_side)
            item_emb = model.item_embed(item_ids.to(device), item_feats=item_side)
        model.train()
        feature_components.user_encoder.train()
        feature_components.item_encoder.train()
        return user_emb, item_emb

    latest_user_emb = None
    latest_item_emb = None
    best_user_emb = None
    best_item_emb = None
    best_score: Optional[float] = None
    saved_states: Dict[str, dict] = {}
    patience_counter = 0

    def debug_similarity(stage_name: str, epoch_idx: int) -> None:
        sample_n = min(config.debug_similarity_samples, len(train_df))
        if sample_n <= 0:
            return

        sample = train_df.sample(sample_n, random_state=(epoch_idx + 1) * (len(stage_name) + 17))
        user_ids = torch.as_tensor(sample["user_idx"].to_numpy(), dtype=torch.long, device=device)
        pos_items = torch.as_tensor(sample["item_idx"].to_numpy(), dtype=torch.long, device=device)

        was_model_training = model.training
        was_user_enc_training = feature_components.user_encoder.training
        was_item_enc_training = feature_components.item_encoder.training
        title_proj = feature_components.item_title_proj
        was_title_proj_training = title_proj.training if title_proj is not None else False

        model.eval()
        feature_components.user_encoder.eval()
        feature_components.item_encoder.eval()
        if title_proj is not None:
            title_proj.eval()

        with torch.no_grad():
            user_feats = encode_cached_batch(
                feature_components.user_feature_cache,
                feature_components.user_encoder,
                user_ids,
                device,
            )
            pos_feats = encode_cached_batch(
                feature_components.item_feature_cache,
                feature_components.item_encoder,
                pos_items,
                device,
            )
            if feature_components.item_title_dim > 0 and title_proj is not None:
                pos_proj = title_proj(feature_components.item_title_embeddings[pos_items])
                pos_feats = torch.cat([pos_feats, pos_proj], dim=1)

            user_vecs, pos_vecs = model(
                user_ids,
                pos_items,
                user_feats=user_feats,
                item_feats=pos_feats,
            )
            pos_scores = torch.sum(user_vecs * pos_vecs, dim=1)

            neg_items = torch.randint(0, num_items, (sample_n,), device=device)
            for _ in range(3):
                mask = neg_items == pos_items
                if not mask.any():
                    break
                neg_items[mask] = torch.randint(0, num_items, (int(mask.sum().item()),), device=device)

            neg_feats = encode_cached_batch(
                feature_components.item_feature_cache,
                feature_components.item_encoder,
                neg_items,
                device,
            )
            if feature_components.item_title_dim > 0 and title_proj is not None:
                neg_proj = title_proj(feature_components.item_title_embeddings[neg_items])
                neg_feats = torch.cat([neg_feats, neg_proj], dim=1)

            neg_vecs = model.item_embed(neg_items, item_feats=neg_feats)
            neg_scores = torch.sum(user_vecs * neg_vecs, dim=1)

            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()
            gap = pos_mean - neg_mean
            hit_ratio = torch.mean((pos_scores > neg_scores).float()).item()

        if was_model_training:
            model.train()
        if was_user_enc_training:
            feature_components.user_encoder.train()
        if was_item_enc_training:
            feature_components.item_encoder.train()
        if title_proj is not None and was_title_proj_training:
            title_proj.train()

        print(
            f"[RecallDebug][{stage_name}] epoch={epoch_idx + 1} pos={pos_mean:.4f} neg={neg_mean:.4f} gap={gap:.4f} hit>{hit_ratio:.3f}"
        )

    def run_training_stage(data_loader: DataLoader, num_epochs: int, stage_name: str) -> None:
        nonlocal best_score, best_user_emb, best_item_emb, latest_user_emb, latest_item_emb, saved_states, patience_counter, k

        if num_epochs <= 0:
            return

        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            encoder_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            epoch_start = time.time()

            for batch in data_loader:
                user_ids = batch["user_id"].to(device)
                pos_items = batch["pos_item"].to(device)

                t0 = time.time()
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
                encoder_time += time.time() - t0

                t0 = time.time()
                optimizer.zero_grad()
                user_out, pos_out = model(
                    user_ids,
                    pos_items,
                    user_feats=user_feats,
                    item_feats=item_feats,
                )
                forward_time += time.time() - t0

                neg_item_tensors = []
                for neg_key in ("easy_neg_items", "tail_neg_items", "hard_neg_items"):
                    if neg_key not in batch:
                        continue
                    neg_tensor = batch[neg_key]
                    if not isinstance(neg_tensor, torch.Tensor) or neg_tensor.numel() == 0:
                        continue
                    neg_tensor = neg_tensor.to(device)
                    mask = neg_tensor >= 0
                    if mask.any():
                        neg_item_tensors.append(neg_tensor[mask])

                if neg_item_tensors:
                    timer_start = time.perf_counter()
                    neg_item_ids = torch.unique(torch.cat(neg_item_tensors, dim=0))
                    if neg_item_ids.numel() > 0:
                        if pos_items.numel() > 0:
                            pos_unique = pos_items.unique()
                            if pos_unique.numel() > 0:
                                if hasattr(torch, "isin"):
                                    keep_mask = ~torch.isin(neg_item_ids, pos_unique)
                                else:
                                    cpu_mask = ~torch.as_tensor(
                                        np.isin(
                                            neg_item_ids.detach().cpu().numpy(),
                                            pos_unique.detach().cpu().numpy(),
                                        ),
                                        device=neg_item_ids.device,
                                    )
                                    keep_mask = cpu_mask
                                neg_item_ids = neg_item_ids[keep_mask]
                        if neg_item_ids.numel() > 0:
                            neg_feats = encode_cached_batch(
                                feature_components.item_feature_cache,
                                feature_components.item_encoder,
                                neg_item_ids,
                                device,
                            )
                            if (
                                feature_components.item_title_dim > 0
                                and feature_components.item_title_proj is not None
                            ):
                                projected_titles = feature_components.item_title_proj(
                                    feature_components.item_title_embeddings[neg_item_ids]
                                )
                                neg_feats = torch.cat([neg_feats, projected_titles], dim=1)
                            neg_emb = model.item_embed(neg_item_ids, item_feats=neg_feats)
                            pos_out = torch.cat([pos_out, neg_emb], dim=0)
                    _ = time.perf_counter() - timer_start

                t0 = time.time()
                loss = loss_fn(user_out, pos_out)
                total_candidates = pos_out.shape[0]
                total_users = user_out.shape[0]
                num_negs_per_pos = total_candidates - 1  # exclude its own positive

                # print(f"[Debug] Each positive is trained against {num_negs_per_pos} negatives "
                #     f"(in-batch + sampled)")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                backward_time += time.time() - t0

            latest_user_emb, latest_item_emb = export_embeddings()
            val_metrics = evaluate_filtered_faiss(
                latest_user_emb,
                latest_item_emb,
                val_users,
                val_pairs,
                user_consumed,
                num_items,
                k=k,
            )
            val_recall = val_metrics.get("recall@k", 0.0)
            epoch_loss = total_loss / max(len(data_loader), 1)
            print(
                f"[Recall] {stage_name} Epoch {epoch + 1}/{num_epochs}, Loss = {epoch_loss:.4f}, "
                f"val_recall@k={val_recall:.4f}, val_ndcg@k={val_metrics.get('ndcg@k', 0.0):.4f}"
            )
            epoch_time = time.time() - epoch_start
            print(
                f"[Timing][Recall][{stage_name}] Epoch {epoch + 1} took {epoch_time:.2f}s "
                f"(encode {encoder_time:.2f}s | forward {forward_time:.2f}s | backward {backward_time:.2f}s)"
            )

            if config.debug_similarity_samples > 0:
                debug_similarity(stage_name, epoch)

            improved = best_score is None or val_recall > best_score + 1e-5
            if improved:
                best_score = val_recall
                best_user_emb = latest_user_emb.detach().clone()
                best_item_emb = latest_item_emb.detach().clone()
                saved_states = {
                    "model": copy.deepcopy(model.state_dict()),
                    "user_encoder": copy.deepcopy(feature_components.user_encoder.state_dict()),
                    "item_encoder": copy.deepcopy(feature_components.item_encoder.state_dict()),
                }
                if feature_components.item_title_proj is not None:
                    saved_states["title_proj"] = copy.deepcopy(
                        feature_components.item_title_proj.state_dict()
                    )
                patience_counter = 0
            else:
                patience_counter += 1
                if (
                    config.early_stop_patience is not None
                    and patience_counter >= config.early_stop_patience
                ):
                    print(
                        f"[Recall] Early stopping on {stage_name} at epoch {epoch + 1}; "
                        f"best val_recall@100={best_score if best_score is not None else 0.0:.4f}"
                    )
                    break
    print("Starting warm-up training stage...")
    run_training_stage(warmup_loader, config.warmup_epochs, "Warm-up")

    warmup_best = {
        "score": best_score,
        "user_emb": best_user_emb.detach().clone() if best_user_emb is not None else None,
        "item_emb": best_item_emb.detach().clone() if best_item_emb is not None else None,
        "states": copy.deepcopy(saved_states) if saved_states else {},
    }

    hard_stage_ready = (
        config.hard_neg_epochs > 0
        and config.hard_neg_samples > 0
        and config.hard_neg_k > 0
        and best_item_emb is not None
    )

    if hard_stage_ready:
        if saved_states:
            model.load_state_dict(saved_states["model"])
            feature_components.user_encoder.load_state_dict(saved_states["user_encoder"])
            feature_components.item_encoder.load_state_dict(saved_states["item_encoder"])
            if (
                feature_components.item_title_proj is not None
                and "title_proj" in saved_states
            ):
                feature_components.item_title_proj.load_state_dict(saved_states["title_proj"])

        hard_item_emb = best_item_emb.detach().cpu()

        # reset best tracking for the hard-negative phase so early stop compares within stage
        best_score = None
        best_user_emb = None
        best_item_emb = None
        saved_states = {}
        patience_counter = 0

        hard_dataset = RecallDataset(
            train_df,
            feature_components.user_store,
            feature_components.item_store,
            item_emb=hard_item_emb,
            easy_neg_samples=config.easy_neg_samples,
            hard_neg_k=config.hard_neg_k,
            hard_neg_samples=config.hard_neg_samples,
            tail_neg_samples=config.tail_neg_samples,
            tail_sampling_alpha=config.tail_sampling_alpha,
            tail_sampling_smoothing=config.tail_sampling_smoothing,
            num_items=num_items,
        )

        hard_loader = DataLoader(
            hard_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print("Starting hard-negative training stage...")
        run_training_stage(hard_loader, config.hard_neg_epochs, "Hard-Neg")

        if (
            best_score is None
            or (warmup_best["score"] is not None and best_score < warmup_best["score"])
        ):
            best_score = warmup_best["score"]
            best_user_emb = (
                warmup_best["user_emb"].detach().clone()
                if warmup_best["user_emb"] is not None
                else best_user_emb
            )
            best_item_emb = (
                warmup_best["item_emb"].detach().clone()
                if warmup_best["item_emb"] is not None
                else best_item_emb
            )
            saved_states = copy.deepcopy(warmup_best["states"]) if warmup_best["states"] else {}

    if best_user_emb is not None and best_item_emb is not None:
        model.load_state_dict(saved_states["model"])
        feature_components.user_encoder.load_state_dict(saved_states["user_encoder"])
        feature_components.item_encoder.load_state_dict(saved_states["item_encoder"])
        if feature_components.item_title_proj is not None and "title_proj" in saved_states:
            feature_components.item_title_proj.load_state_dict(saved_states["title_proj"])
        user_emb, item_emb = best_user_emb, best_item_emb
    else:
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
