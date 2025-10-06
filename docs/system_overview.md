# Project Overview

This document captures the current design of the recommendation pipeline, split into feature processing, recall, ranking, and evaluation. The goal is to highlight the functionality of each stage along with the most relevant configuration knobs.

## Feature Layer

Feature construction happens in `src/pipeline/feature_pipeline.py`, which orchestrates the user/item preprocessing helpers and builds `FeatureStore` instances for both sides.

### Item features (`src/features/item_features.py`)

* **Rating statistics** – total interactions, average rating, and variance per item. Missing values are filled with global means (variance defaults to `0`).
* **Release year** – parsed from the title and normalized; `FeatureStore` later buckets the value for embeddings.
* **Popularity signals**
  * `item_interaction_count` – raw counts over training data.
  * `item_interaction_log` – log‑scaled popularity used in bucketing.
  * `item_popularity_rank` – dense rank (0..1) to capture popularity relative to the catalog.
  * `item_recency_days` – days since last interaction, filtered from raw timestamps.
* **Genre richness** – number of genres attached to the movie.
* **Title semantics** – `item_title_tokens` is a lightweight bag‑of‑words extracted from the title (parenthetical year removed, minimum token length 3). These feed `FeatureStore`’s multi-categorical pipeline.
* **Sentence embeddings** – titles are processed through `load_or_compute_title_embeddings`, cached in `data/cache/item_title_embeddings.pt`, and optionally projected to a smaller dimensionality via a learnable linear layer (`item_title_proj`).

All item features are stored in a `FeatureStore` with:
* No dense "numeric" vectors (they are bucketized instead).
* Bucketized popularity, recency, rating stats, etc.
* Multi-categorical inputs for genres and title tokens.

### User features (`src/features/user_features.py`)

User features capture both aggregate preferences and short-term behaviour.

* **Aggregated statistics** – total ratings, mean rating, rating variance, recency (days since last interaction), weighted average release year.
* **Preference buckets** – `temporal_preference` classifies users as `prefers_classics`, `prefers_recent`, or `balanced` using a configurable margin.
* **Histories** – `watched_items` keeps up to the last 100 items; `recent_items` holds the 20 most recent IDs (these feed the multi-categorical feature channel).
* **Genres**
  * `favorite_genres` – top 3 genres by frequency (ties resolved by rating sum).
  * `recent_genres` – genres from the most recent 20 interactions.
  * `user_genre_entropy` – entropy of the genre distribution (higher means broader taste).
* **Recency-weighted ratings** – `user_recent_rating` averages ratings with linear weights (more recent interactions receive a higher weight).

A `FeatureStore` sits on top with:
* Bucketized numerical columns (rating stats, recency, entropy, recent rating).
* Single categorical column (`temporal_preference`).
* Multi-categorical columns for histories and genres (per-column max lengths controlled by `user_multi_max`).

### Encoding

`FeatureEncoder` modules (32‑dim embeddings + projection) consume the `FeatureStore` batches and return dense feature vectors for both user and item towers. Title embeddings are optionally projected to 64‑dim and concatenated with encoded item features.

## Recall Stage

Recall is responsible for generating a broad candidate set via a two-stage process:

1. **Two-Tower training** (`src/pipeline/recall_training.py`)
2. **Candidate synthesis** (`src/pipeline/candidate_builder.py`)

### Training dataset (`src/dataset/recall_dataset.py`)

`RecallDataset` draws samples from the training ratings and attaches three types of negatives:

* **Easy negatives** – uniform random catalog draws avoiding consumed items.
* **Tail negatives** – random draws weighted by `count^-alpha` (`tail_sampling_alpha`), steering towards the long tail. Controlled via `NegativeConfig.tail` and smoothing options.
* **Hard negatives** – nearest neighbours in the item embedding space (FAISS top‑`hard_neg_k`), excluding consumed items. Activated in the hard-negative phase.

All negatives are returned as tensors (`easy_neg_items`, `tail_neg_items`, `hard_neg_items`), and each mini-batch still leverages in-batch negatives through the `InBatchSoftmaxLoss` (InfoNCE).

### Training schedule

`train_two_tower_model` runs in two phases:

1. **Warm-up**
   * Only easy + tail negatives.
   * Tracks the best validation recall (FAISS evaluation with train interactions filtered).
   * Logs debug stats (`[RecallDebug]`) showing mean positive score, mean random negative score, gap, and hit ratio.
2. **Hard-negative phase** (optional)
   * Resets early-stopping counters so hard-neg improvements are evaluated independently.
   * Rebuilds the dataset with `NegativeConfig` enabling FAISS hard negatives using the best warm-up item embeddings.
   * Falls back to the warm-up checkpoint if the hard stage fails to surpass the best warm-up metric.

The model itself is `TwoTowerModel` (user/item ID embeddings + feature MLPs, layer norm, L2-normalized outputs). Training uses `InBatchSoftmaxLoss`, Adam optimizer, and evaluation via filtered FAISS recall@100.

### Candidate builder

`build_candidate_sources` returns `CandidateSources` containing:
* FAISS embedding index for on-the-fly nearest neighbours.
* Covisitation and item-CF indices/user candidate maps.
* Popular items and per-user popularity candidates.
* Genre heuristics – maps of genre → popular items, plus per-user top genres.

`build_hybrid_candidates` blends multiple lists with configurable weights:
1. **FAISS top-K (after removing consumed items)**.
2. **Covisitation candidates**.
3. **Item-based collaborative filtering**.
4. **Popularity-based list**.
5. **Genre/topical boost** – append genre-aligned items per user (capped and weighted). Weights default to a modest value but can be tuned.

The merged candidate map becomes input for both ranking and evaluation stages.

## Ranking Stage

Ranking uses a pointwise feature-rich model to reorder recall candidates. Current options (`RankTrainingConfig.model_type`) include: `dcn`, `dcn_din`, `deepfm`, `din`, `sasrec`. They vary by architecture:

* **DCN / DCN-DIN** – cross networks over dense features; `dcn_din` also conditions on history via attention.
* **DeepFM** – combines FM-style second-order interactions with a DNN.
* **DIN** – attention over history items conditioned on the target item.
* **SASRec** – transformer-based encoder over user history (requires `history_max` set > 0).

### Ranking dataset (`src/dataset/rank_dataset.py`)

`RankDataset` now returns multi-task training examples `(user_id, pos_item, neg_items, hist_items, pos_rating)`:
* Positives come from chronological interactions (or `positive_pairs`) with per-user history snapshots trimmed to `max_history`.
* Negatives are sampled on-demand per access: a configurable fraction from the recall candidate map (`candidate_neg_ratio`) and the rest from tail-random catalog draws.
* Ratings (if available in the source data) are preserved for the positive item to support satisfaction modeling.

### Training (`src/pipeline/rank_training.py`)

* **Optimizer & scheduler** – Adam with cosine annealing (`T_max = epochs`).
* **Losses**
  * Pairwise BPR (`bpr_loss`) on ranking scores.
  * Margin regulariser (`max(0, margin - (pos_mean - neg_mean))`).
  * Optional rating regression loss (MSE against the survivor rating) weighted by `rating_loss_weight`.
* **Models** – In addition to `dcn/dcn_din/deepfm/din/sasrec`, the pipeline supports an MMOE ranker (`model_type="mmoe"`) that shares expert towers but feeds target-specific gating into:
  * A ranking head (either a simple MLP or wrapped DeepFM/DCN backbone).
  * A rating head (always MLP) predicting 0–5 satisfaction scores.
* **Batch flow** – Per batch we score positives and their sampled negatives using the chosen model, compute the multi-task losses, and log `[RankDebug]` (mean positive/negative scores, gap, hit ratio).
* **Evaluation** – `evaluate_ranker_with_candidates` detects tuple outputs (rank + rating) and uses the ranking head for recall/NDCG/GAUC. Skip modes (`RANK_SKIP_MISSING`) remain available to isolate pure re-ranking quality.

## Evaluation Layer

`src/evaluation.py` offers:

* `evaluate_candidates` – evaluates raw candidate pools (recall@K, NDCG@K). Supports `skip_missing` (useful when recall coverage matters vs pure ranking).
* `evaluate_filtered_faiss` – recall evaluation on two-tower embeddings with training interactions filtered out.
* `evaluate_ranker_with_candidates` – scores candidate lists with a given ranker and computes recall@K, NDCG@K, and GAUC. Accepts history-aware models, user/item feature matrices, and supports `skip_missing_candidates` to ignore users whose gold item is absent.

At a high level, the evaluation strategy enforces that recall coverage limits ranking performance; the optional skip flag helps isolate ranker quality.

---

This document should provide enough context to extend or debug each stage. For any changes, keep an eye on the diagnostic logs (`[RecallDebug]`, `[RankDebug]`) to ensure improvements translate across the pipeline.
