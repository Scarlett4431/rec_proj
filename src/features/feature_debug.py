from typing import Optional

import torch

from src.features.feature_utils import encode_cached_batch


def debug_feature_usage(
    name: str,
    encoder,
    cache,
    entity_count: int,
    device: torch.device,
    extra_tensor: Optional[torch.Tensor] = None,
    sample_size: int = 1024,
) -> None:
    """Print basic diagnostics about feature coverage for a given encoder."""

    if entity_count <= 0:
        print(f"[Debug] {name}: no entities available")
        return

    if getattr(encoder, "out_dim", 0) == 0:
        print(f"[Debug] {name}: encoder output dimension is 0")
        return

    sample_size = min(sample_size, entity_count)
    sample_ids = torch.randperm(entity_count)[:sample_size]

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        encoded = encode_cached_batch(cache, encoder, sample_ids, device)
    if was_training:
        encoder.train()

    combined = encoded
    if extra_tensor is not None and extra_tensor.numel() > 0:
        extra_chunk = extra_tensor[sample_ids.to(extra_tensor.device)]
        combined = torch.cat([combined, extra_chunk.to(combined.device)], dim=1)

    if combined.numel() == 0:
        print(f"[Debug] {name}: combined feature tensor empty")
        return

    zero_mask = combined.abs().sum(dim=0) == 0
    zero_cols = zero_mask.nonzero(as_tuple=False).flatten().tolist()
    print(
        f"[Debug] {name}: sample={sample_size}, dim={combined.size(1)}, zero_dims={len(zero_cols)}"
    )
    if zero_cols:
        preview = zero_cols[:10]
        suffix = "..." if len(zero_cols) > 10 else ""
        print(f"           zero-dim indices: {preview}{suffix}")
