# Recommendation System Releases

This repository tracks incremental versions of our feature Recall Rank pipeline. Each release captures the major changes, models, and headline metrics.

---

## Version 1.0
- **Features**: User/item inputs were limited to binned numerical interaction counts.
- **Recall**: Two-tower model with two-stage training (in-batch negatives, followed by FAISS-mined hard negatives via `InBatchSoftmaxLoss`).
- **Ranking**: Simple feed-forward ranker trained on the recall top‑K with BCE loss.
- **Metrics**:
  - FAISS Recall @200: `recall=0.22, ndcg=0.04`
  - Ranker @10: `recall=0.025, ndcg=0.01`

## Version 1.1
- **Features**: Added multi-categorical embeddings (`watched_items`, `item_genres`).
- **Metrics**:
  - FAISS Recall @200: `recall=0.28, ndcg=0.055`
  - Ranker @10: `recall=0.032, ndcg=0.015`

## Version 1.2
- **User features**: Added top-3 favorite genres and temporal preference buckets.
- **Recall**: Introduced multiple candidate streams (co-visitation, item-CF, popularity) with weighted merge.
- **Metrics**:
  - Hybrid Recall @100: `recall=0.32, ndcg=0.065`
  - Ranker @10: `recall=0.041, ndcg=0.021`

## Version 1.3
- **Recall enhancements**: Two-tower embeddings L2-normalized; easy-negative sampler mixes random negatives with cache; InfoNCE temperature set to 0.1.
- **Metrics**: Hybrid Recall @100: `recall=0.35, ndcg=0.075`

## Version 1.4
- **Evaluation**: Filter training interactions out of FAISS search.
- **Hybrid merge**: Weighted scoring instead of integer quotas.
- **Metrics**:
  - Two-tower only @100: `recall=0.42, ndcg=0.11`
  - Hybrid Recall @100: `recall=0.45, ndcg=0.12`
  - Ranker @10: `recall=0.055, ndcg=0.028`

## Version 1.5
- **Recall**: Removed hard-negative fine-tuning (hurt validation).
- **Ranking**: Replaced MLP with DeepFM/DCN.
- **Feature**: Added title embeddings (MiniLM) projected to 64d.
- **Metrics**: Hybrid Recall @100: `recall=0.48, ndcg=0.135`

## Version 1.6
- **Ranking**: Added DIN (attention over history) and SASRec variants.
- **Features**: Hybrid binning (linear + log) and rating variance.
- **Evaluation**: Added hit rate, coverage, GAUC.
- **Metrics**: Recall @100: `recall=0.52, ndcg=0.15`

## Version 1.7
- **Debugging**: Added RECALL/RANK debug env vars to inspect histories and recency scores; `[RecallDebug]` logs pos vs random neg gap.
- **Recall**: Multi-stage negative sampling (easy/tail, later hard) + genre heuristics for candidates.
- **Ranking**: Overhauled negative sampling (was purely random; later migrated to candidate-aware sampling in v1.8).
- **Metrics**: Hybrid Recall @100: `recall=0.56, ndcg=0.17`

## Version 1.8
- **Ranking**: Switched from BCE to BPR loss, mixing random and recall-based negatives.
- **Current status**:
  - Ranker @10 (with skip): `recall=0.24, ndcg=0.14, gauc=0.66`

## Version 1.9
- **Ranking dataset**: Samples negatives on-demand (candidate + random mix) and retains per-interaction ratings for multi-task training.
- **Optimizer**: Cosine-annealed learning rate, BPR + margin regularisation, optional rating regression loss.
- **Models**: Added MMOE ranker (`model_type="mmoe"`) that can wrap MLP, DeepFM, or DCN heads while sharing experts across ranking/rating tasks.
- **Evaluation**: Updated to consume tuple outputs (rank + rating) seamlessly.
- **Metrics**: (example) Ranker @10 (skip-missing) `recall=0.6, ndcg=0.34, gauc=0.72` with multi-task training.

---

## Roadmap / TODO
- Multi-task towers (MMOE/PLE/ESSM) for engagement vs satisfaction objectives.
- Explore additional multi-task architectures (PLE/ESSM) and revisit loss calibration once hard-negative tuning settles.

For deeper architectural notes, see `docs/system_overview.md`.
