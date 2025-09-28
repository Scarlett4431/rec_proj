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

