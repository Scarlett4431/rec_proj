import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


CACHE_FILENAMES = {
    "user_embeddings": "user_embeddings.pt",
    "item_embeddings": "item_embeddings.pt",
    "user_features": "user_features.pt",
    "item_features": "item_features.pt",
    "user_candidates": "user_candidates.json",
}


def save_recall_cache(
    cache_dir: Path,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    user_feat_matrix: torch.Tensor,
    item_feat_matrix: torch.Tensor,
    user_candidates: Dict[int, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(user_embeddings.cpu(), cache_dir / CACHE_FILENAMES["user_embeddings"])
    torch.save(item_embeddings.cpu(), cache_dir / CACHE_FILENAMES["item_embeddings"])
    torch.save(user_feat_matrix.cpu(), cache_dir / CACHE_FILENAMES["user_features"])
    torch.save(item_feat_matrix.cpu(), cache_dir / CACHE_FILENAMES["item_features"])
    with open(cache_dir / CACHE_FILENAMES["user_candidates"], "w") as f:
        json.dump({str(k): v for k, v in user_candidates.items()}, f)


def load_recall_cache(cache_dir: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Any]]]:
    try:
        user_emb = torch.load(cache_dir / CACHE_FILENAMES["user_embeddings"])
        item_emb = torch.load(cache_dir / CACHE_FILENAMES["item_embeddings"])
        user_feats = torch.load(cache_dir / CACHE_FILENAMES["user_features"])
        item_feats = torch.load(cache_dir / CACHE_FILENAMES["item_features"])
        with open(cache_dir / CACHE_FILENAMES["user_candidates"], "r") as f:
            user_candidates = {int(k): v for k, v in json.load(f).items()}
        return user_emb, item_emb, user_feats, item_feats, user_candidates
    except FileNotFoundError:
        return None
