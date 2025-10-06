# Recommendation System Releases

This repository tracks incremental versions of our feature  Recall  Rank pipeline. Each release captures the major changes, models, and headline metrics.

---

## Version 1.0
- **Features**: User/item inputs were limited to binned numerical interaction counts.
- **Recall**: Two-tower model with two-stage training (in-batch negatives, followed by FAISS-mined hard negatives via `InBatchSoftmaxLoss`).
- **Ranking**: Simple feed-forward ranker trained on the recall top‑K with BCE loss.
- **Metrics**:
  - FAISS Recall @200: `recall=0.1672, ndcg=0.0302`
  - Ranker @10: `recall=0.0153, ndcg=0.0071`

## Version 1.1
- **Features**: Added multi-categorical embeddings (`watched_items`, `item_genres`).
- **Metrics**:
  - FAISS Recall @200: `recall=0.2564, ndcg=0.0465`
  - Ranker @10: `recall=0.0264, ndcg=0.0119`

## Version 1.2
- **User features**: Added top-3 favorite genres and temporal preference buckets.
- **Recall**: Introduced multiple candidate streams (co-visitation, item-CF, popularity) with weighted merge.
- **Metrics**:
  - Hybrid Recall @100: `recall=0.2399, ndcg=0.0466`
  - Ranker @10: `recall=0.0303, ndcg=0.0135`

## Version 1.3
- **Recall enhancements**: Two-tower embeddings L2-normalized; easy-negative sampler mixes random negatives with cache; InfoNCE temperature set to 0.1.
- **Metrics**: Hybrid Recall @100: `recall=0.2625, ndcg=0.0550`

## Version 1.4
- **Evaluation**: Filter training interactions out of FAISS search.
- **Hybrid merge**: Weighted scoring instead of integer quotas.
- **Metrics**:
  - Two-tower only @100: `recall=0.3557, ndcg=0.0876`
  - Hybrid Recall @100: `recall=0.3861, ndcg=0.0980`
  - Ranker @10: `recall=0.0441, ndcg=0.0201`

## Version 1.5
- **Recall**: Removed hard-negative fine-tuning (hurt validation).
- **Ranking**: Replaced MLP with DeepFM/DCN; added title embeddings (MiniLM) projected to 64d.
- **Metrics**: Hybrid Recall @100: `recall=0.4076, ndcg=0.1039`

## Version 1.6
- **Ranking**: Added DIN (attention over history) and SASRec variants.
- **Features**: Hybrid binning (linear + log) and rating variance.
- **Evaluation**: Added hit rate, coverage, GAUC.
- **Metrics**: Recall @100: `recall=0.4191, ndcg=0.1076`

## Version 1.7
- **Debugging**: Added RECALL/RANK debug env vars to inspect histories and recency scores; `[RecallDebug]` logs pos vs random neg gap.
- **Recall**: Multi-stage negative sampling (easy/tail, later hard) + genre heuristics for candidates.
- **Ranking**: Overhauled negative sampling (was purely random; later migrated to candidate-aware sampling in v1.8).
- **Metrics**: Hybrid Recall @100: `recall=0.4329, ndcg=0.1102`

## Version 1.8
- **Ranking**: Switched from BCE to BPR loss, mixing random and recall-based negatives.
- **Current status**:
  - Ranker @10 (with skip): `recall=0.1438, ndcg=0.0712, gauc=0.6195`
  - Ranker @10 (pure re-ranking): `recall=0.1200, ndcg=0.0587, gauc=0.6513`

## Version 1.9
- **Ranking dataset**: Samples negatives on-demand (candidate + random mix) and retains per-interaction ratings for multi-task training.
- **Optimizer**: Cosine-annealed learning rate, BPR + margin regularisation, optional rating regression loss.
- **Models**: Added MMOE ranker (`model_type="mmoe"`) that can wrap MLP, DeepFM, or DCN heads while sharing experts across ranking/rating tasks.
- **Evaluation**: Updated to consume tuple outputs (rank + rating) seamlessly.
- **Metrics**: (example) Ranker @10 (skip-missing) `recall=0.14, ndcg=0.07, gauc≈0.62` with multi-task training.

---

## Roadmap / TODO
- Multi-task towers (MMOE/PLE/ESSM) for engagement vs satisfaction objectives.
- Explore additional multi-task architectures (PLE/ESSM) and revisit loss calibration once hard-negative tuning settles.

For deeper architectural notes, see `docs/system_overview.md`.
