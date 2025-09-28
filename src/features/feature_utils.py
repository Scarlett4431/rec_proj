from typing import Any, Dict

import torch


def move_to_device(struct, device):
    if isinstance(struct, torch.Tensor):
        return struct.to(device)
    if isinstance(struct, dict):
        return {k: move_to_device(v, device) for k, v in struct.items()}
    return struct


def encode_features(encoder, feature_batch, device):
    feature_batch = move_to_device(feature_batch, device)
    return encoder(feature_batch)


def encode_store_batch(store, encoder, ids, device, max_lengths):
    if isinstance(ids, torch.Tensor):
        id_list = ids.detach().cpu().tolist()
    else:
        id_list = ids
    batch = store.get_batch(id_list, max_multi_lengths=max_lengths)
    return encode_features(encoder, batch, device)


def encode_all_entities(store, encoder, count, device, max_lengths):
    ids = list(range(count))
    return encode_store_batch(store, encoder, ids, device, max_lengths)


def _ensure_long_tensor(ids) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.detach().cpu().long()
    return torch.as_tensor(ids, dtype=torch.long)


def gather_cached_features(cache: Dict[str, Any], ids) -> Dict[str, Any]:
    idx = _ensure_long_tensor(ids)
    result: Dict[str, Any] = {}

    numeric = cache.get("numeric")
    if numeric is not None:
        result["numeric"] = numeric.index_select(0, idx)
    else:
        result["numeric"] = torch.zeros((len(idx), 0), dtype=torch.float32)

    result["categorical"] = {
        col: tensor.index_select(0, idx)
        for col, tensor in cache.get("categorical", {}).items()
    }

    result["bucket"] = {
        col: tensor.index_select(0, idx)
        for col, tensor in cache.get("bucket", {}).items()
    }

    result["multi_categorical"] = {
        col: tensor.index_select(0, idx)
        for col, tensor in cache.get("multi_categorical", {}).items()
    }

    result["multi_lengths"] = {
        col: tensor.index_select(0, idx)
        for col, tensor in cache.get("multi_lengths", {}).items()
    }

    return result


def encode_cached_batch(cache: Dict[str, Any], encoder, ids, device):
    batch = gather_cached_features(cache, ids)
    return encode_features(encoder, batch, device)
