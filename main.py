import os
from pathlib import Path

import torch
from src.features.feature_utils import encode_features
from src.features.feature_debug import debug_feature_usage
from src.evaluation import evaluate_candidates
from src.pipeline.data_prep import load_and_prepare_data
from src.pipeline.feature_pipeline import build_feature_components
from src.pipeline.recall_training import RecallTrainingConfig, train_two_tower_model
from src.pipeline.candidate_builder import build_candidate_sources, build_hybrid_candidates
from src.pipeline.rank_training import RankTrainingConfig, train_ranker_model
from src.pipeline.cache_utils import save_recall_cache, load_recall_cache
from src.recall.debug_utils import inspect_recall_result
from src.rank.debug_utils import inspect_rank_result
from src.utils import build_item_name_lookup


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import time
    candidate_k = 100
    start_time = time.time()
    data = load_and_prepare_data("data/ml-1m")
    print(f"[Timing] Data prep took {time.time() - start_time:.2f}s")
    step_time = time.time()
    item_name_lookup = build_item_name_lookup(data.movies)

    cache_dir = Path("cache/recall_outputs")
    step_time = time.time()
    cache_data = load_recall_cache(cache_dir)
    print(f"[Timing] Cache lookup took {time.time() - step_time:.2f}s")
    feature_components = None
    if False:
    # if cache_data is not None:
        print("Loaded recall artifacts from cache.")
        (
            user_embeddings,
            item_embeddings,
            user_feat_matrix_cpu,
            item_feat_matrix_cpu,
            user_candidates,
        ) = cache_data
    else:
        feature_components = build_feature_components(
            data.train,
            data.movies,
            num_users=data.num_users,
            num_items=data.num_items,
            device=device,
        )
        print(f"[Timing] Feature components built in {time.time() - step_time:.2f}s")

        debug_feature_usage(
            "user",
            feature_components.user_encoder,
            feature_components.user_feature_cache,
            data.num_users,
            device,
        )
        debug_feature_usage(
            "item",
            feature_components.item_encoder,
            feature_components.item_feature_cache,
            data.num_items,
            device,
            extra_tensor=feature_components.item_title_embeddings,
        )

        step_time = time.time()
        recall_outputs = train_two_tower_model(
            data.train,
            data.num_users,
            data.num_items,
            data.user_consumed,
            data.val_users,
            data.val_pairs,
            feature_components,
            device,
            config=RecallTrainingConfig(),
        )
        print(f"[Timing] Recall training took {time.time() - step_time:.2f}s")

        user_embeddings = recall_outputs.user_embeddings
        item_embeddings = recall_outputs.item_embeddings

        step_time = time.time()
        candidate_sources = build_candidate_sources(
            data.train,
            data.movies,
            data.num_users,
            data.user_consumed,
        )
        print(f"[Timing] Candidate sources built in {time.time() - step_time:.2f}s")

        step_time = time.time()
        user_candidates = build_hybrid_candidates(
            user_embeddings,
            item_embeddings,
            data.user_consumed,
            candidate_sources,
            candidate_k=candidate_k,
            faiss_weight=1,
            # covis_weight=0.4,
            # itemcf_weight=0.4,
            # popular_weight=0,
            # genre_weight=0.2,
        )
        print(f"[Timing] Hybrid candidate building took {time.time() - step_time:.2f}s")

        recall_results = evaluate_candidates(user_candidates, data.test_pairs, k=candidate_k)
        print("Hybrid Recall Eval (after warm-up):", recall_results)

        feature_components.user_encoder.eval()
        feature_components.item_encoder.eval()
        for param in feature_components.user_encoder.parameters():
            param.requires_grad_(False)
        for param in feature_components.item_encoder.parameters():
            param.requires_grad_(False)

        user_feat_matrix_cpu = encode_features(
            feature_components.user_encoder,
            feature_components.user_feature_cache,
            device,
        ).detach().cpu()
        item_feat_encoded = encode_features(
            feature_components.item_encoder,
            feature_components.item_feature_cache,
            device,
        ).detach().cpu()
        if feature_components.item_title_dim > 0 and feature_components.item_title_proj is not None:
            projected_titles = feature_components.item_title_proj(
                feature_components.item_title_embeddings
            ).detach().cpu()
            item_feat_matrix_cpu = torch.cat(
                [
                    item_feat_encoded,
                    projected_titles,
                ],
                dim=1,
            )
        else:
            item_feat_matrix_cpu = item_feat_encoded

        cache_save_time = time.time()
        save_recall_cache(
            cache_dir,
            user_embeddings,
            item_embeddings,
            user_feat_matrix_cpu,
            item_feat_matrix_cpu,
            user_candidates,
        )
        print(
            f"Saved recall artifacts to cache (took {time.time() - cache_save_time:.2f}s)."
        )

    debug_user_env = os.environ.get("RECALL_DEBUG_USER")
    if debug_user_env:
        try:
            debug_users = [int(u.strip()) for u in debug_user_env.split(",") if u.strip()]
        except ValueError:
            print(f"[RecallDebug] Invalid RECALL_DEBUG_USER='{debug_user_env}'")
        else:
            for debug_user in debug_users:
                debug_target = next(
                    (item for user, item in data.test_pairs if user == debug_user),
                    None,
                )
                inspect_recall_result(
                    debug_user,
                    debug_target,
                    user_candidates.get(debug_user, []),
                    item_name_lookup,
                    user_history=data.user_histories.get(debug_user),
                    history_k=10,
                    top_k=20,
                )
    skip_missing_rank_env = os.environ.get("RANK_SKIP_MISSING", "")
    skip_missing_rank_eval = skip_missing_rank_env.lower() in ("1", "true", "yes")
    step_time = time.time()
    rank_outputs = train_ranker_model(
        data.train,
        data.val_pairs,
        data.num_items,
        user_embeddings,
        item_embeddings,
        user_feat_matrix_cpu,
        item_feat_matrix_cpu,
        user_candidates,
        data.test_pairs,
        data.user_histories,
        device,
        config=RankTrainingConfig(),
        skip_missing_eval=skip_missing_rank_eval,
    )
    print(f"[Timing] Rank training took {time.time() - step_time:.2f}s")

    print("Ranker Eval (Hybrid Recall+Rank):", rank_outputs.metrics)
    if skip_missing_rank_eval:
        print("[RankEval] Skipped users without gold candidates (pure ranking view)")

    rank_debug_env = os.environ.get("RANK_DEBUG_USER")
    if rank_debug_env:
        try:
            rank_debug_users = [int(u.strip()) for u in rank_debug_env.split(",") if u.strip()]
        except ValueError:
            print(f"[RankDebug] Invalid RANK_DEBUG_USER='{rank_debug_env}'")
        else:
            for debug_user in rank_debug_users:
                gold_items = [item for user, item in data.test_pairs if user == debug_user]
                inspect_rank_result(
                    rank_outputs.ranker,
                    debug_user,
                    user_candidates.get(debug_user, []),
                    item_name_lookup,
                    user_emb=user_embeddings,
                    item_emb=item_embeddings,
                    user_feat_matrix=user_feat_matrix_cpu,
                    item_feat_matrix=item_feat_matrix_cpu,
                    user_histories=data.user_histories,
                    gold_items=gold_items,
                    max_history=rank_outputs.max_history,
                    history_k=10,
                    top_k=20,
                    device=device,
                )


if __name__ == "__main__":
    main()
