from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.feature_store import FeatureStore
from src.features.feature_encoder import FeatureEncoder
from src.features.title_embeddings import load_or_compute_title_embeddings


@dataclass
class FeatureComponents:
    user_store: FeatureStore
    item_store: FeatureStore
    user_encoder: FeatureEncoder
    item_encoder: FeatureEncoder
    user_feature_cache: Dict[str, torch.Tensor]
    item_feature_cache: Dict[str, torch.Tensor]
    user_multi_max: Dict[str, int]
    item_multi_max: Dict[str, int]
    item_title_embeddings: torch.Tensor
    item_title_dim: int
    item_title_proj: nn.Module | None


def build_feature_components(
    train_df,
    movies_df,
    num_users: int,
    num_items: int,
    device: torch.device,
    title_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    title_cache_path: str = "data/cache/item_title_embeddings.pt",
) -> FeatureComponents:
    user_feats_df = build_user_features(train_df, movies_df)
    item_feats_df = build_item_features(train_df, movies_df)

    user_store = FeatureStore(
        user_feats_df,
        "user_idx",
        numeric_cols=[],
        cat_cols=["temporal_preference"],
        multi_cat_cols=["watched_items", "favorite_genres"],
        bucket_cols=["user_total_ratings", "user_avg_rating", "user_recency_days", "user_avg_release_year"],
        bucket_bins=10,
    )

    item_numeric_cols = ["item_total_ratings", "item_avg_rating", "item_release_year"]
    item_store = FeatureStore(
        item_feats_df,
        "item_idx",
        numeric_cols=[],
        cat_cols=[],
        multi_cat_cols=["item_genres"],
        bucket_cols=item_numeric_cols,
        bucket_bins=10,
    )

    user_encoder = FeatureEncoder(
        numeric_dim=user_store.numeric_dim,
        cat_dims=user_store.cat_dims,
        bucket_dims=user_store.bucket_dims,
        multi_cat_dims=user_store.multi_cat_dims,
        embed_dim=32,
        proj_dim=32,
    ).to(device)

    item_encoder = FeatureEncoder(
        numeric_dim=item_store.numeric_dim,
        cat_dims=item_store.cat_dims,
        bucket_dims=item_store.bucket_dims,
        multi_cat_dims=item_store.multi_cat_dims,
        embed_dim=32,
        proj_dim=32,
    ).to(device)

    user_multi_max = {"watched_items": 50, "favorite_genres": 3}
    item_multi_max = {"item_genres": 5}

    user_feature_cache = user_store.get_batch(list(range(num_users)), max_multi_lengths=user_multi_max)
    item_feature_cache = item_store.get_batch(list(range(num_items)), max_multi_lengths=item_multi_max)

    titles_sorted = (
        movies_df.sort_values("item_idx")["title"].fillna("").astype(str).tolist()
        if not movies_df.empty
        else []
    )
    title_cache = Path(title_cache_path)
    title_embeddings_cpu = load_or_compute_title_embeddings(
        titles_sorted,
        cache_path=title_cache,
        model_name=title_model_name,
        batch_size=256,
        device=torch.device("cpu"),
    )
    item_title_dim = title_embeddings_cpu.shape[1] if title_embeddings_cpu.numel() > 0 else 0
    if item_title_dim > 0 and title_embeddings_cpu.shape[0] != num_items:
        raise ValueError(
            f"Title embedding count {title_embeddings_cpu.shape[0]} does not match num_items {num_items}"
        )
    item_title_embeddings = (
        title_embeddings_cpu.to(device)
        if item_title_dim > 0
        else torch.zeros((num_items, 0), device=device)
    )

    title_proj = None
    if item_title_dim > 0:
        title_proj = nn.Linear(item_title_dim, 64)
        nn.init.xavier_uniform_(title_proj.weight)
        if title_proj.bias is not None:
            nn.init.zeros_(title_proj.bias)
        title_proj.to(device)

    return FeatureComponents(
        user_store=user_store,
        item_store=item_store,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_feature_cache=user_feature_cache,
        item_feature_cache=item_feature_cache,
        user_multi_max=user_multi_max,
        item_multi_max=item_multi_max,
        item_title_embeddings=item_title_embeddings,
        item_title_dim=item_title_dim,
        item_title_proj=title_proj,
    )
