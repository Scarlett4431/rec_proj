import torch
from pathlib import Path
from src.features.feature_utils import encode_features
from src.features.feature_debug import debug_feature_usage
from src.evaluation import evaluate_candidates
from src.pipeline.data_prep import load_and_prepare_data
from src.pipeline.feature_pipeline import build_feature_components
from src.pipeline.recall_training import RecallTrainingConfig, train_two_tower_model
from src.pipeline.candidate_builder import build_candidate_sources, build_hybrid_candidates
from src.pipeline.rank_training import RankTrainingConfig, train_ranker_model
from src.pipeline.cache_utils import save_recall_cache, load_recall_cache


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_and_prepare_data("data/ml-1m")
    feature_components = build_feature_components(
        data.train,
        data.movies,
        num_users=data.num_users,
        num_items=data.num_items,
        device=device,
    )

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

    cache_dir = Path("cache/recall_outputs")
    cache_data = load_recall_cache(cache_dir)
    # if cache_data is not None:
    if False:
        print("Loaded recall artifacts from cache.")
        (
            user_embeddings,
            item_embeddings,
            user_feat_matrix_cpu,
            item_feat_matrix_cpu,
            user_candidates,
        ) = cache_data
    else:
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

        user_embeddings = recall_outputs.user_embeddings
        item_embeddings = recall_outputs.item_embeddings

        candidate_sources = build_candidate_sources(
            data.train,
            data.num_users,
            data.user_consumed,
        )

        user_candidates = build_hybrid_candidates(
            user_embeddings,
            item_embeddings,
            data.user_consumed,
            candidate_sources,
            candidate_k=100,
            faiss_weight=0.7,
            itemcf_weight=0.5,
            popular_weight=0.3,
        )

        recall_results = evaluate_candidates(user_candidates, data.test_pairs, k=100)
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

        save_recall_cache(
            cache_dir,
            user_embeddings,
            item_embeddings,
            user_feat_matrix_cpu,
            item_feat_matrix_cpu,
            user_candidates,
        )
        print("Saved recall artifacts to cache.")
    rank_outputs = train_ranker_model(
        data.train,
        data.val_pairs,
        data.num_items,
        feature_components.user_store,
        feature_components.item_store,
        user_embeddings,
        item_embeddings,
        user_feat_matrix_cpu,
        item_feat_matrix_cpu,
        user_candidates,
        data.test_pairs,
        data.user_histories,
        device,
        config=RankTrainingConfig(),
    )

    print("Ranker Eval (Hybrid Recall+Rank):", rank_outputs.metrics)


if __name__ == "__main__":
    main()
