Version 1.0:
Features: both item and user only include binned numerical features, that is mostly interation based.
Recall: two-tower approach, two stage training. First stage in-batch negative sampling, second stage also use FAISS to mine top-k neighbors as hard negative pool for fine tuning. Loss using InBatchSoftmaxLoss().
Ranking: simple RankerMLP model. Rank train and inference only on the top K items retrieved from the recall stage. Loss using BCELoss().
Result:
FAISS Recall Eval (after hard-neg): {'recall@k': 0.16716358179089544, 'ndcg@k': 0.030157848635290953}, K = 200
Ranker Eval (Recall+Rank): {'recall@k': 0.015286789774147429, 'ndcg@k': 0.0071179959781579} K = 10

Version 1.1:
Add in support for multi-categorical feature embedding, currently pooling inside the FeatureEncoder. Added feature watched_items for user and item_genres for item.
Result: 
FAISS Recall Eval (after hard-neg): {'recall@k': 0.2563781890945473, 'ndcg@k': 0.04650883651231697}, K = 200
FAISS Recall Eval (after hard-neg): {'recall@k': 0.15741870935467733, 'ndcg@k': 0.0336246098628547}, K = 100
Ranker Eval (Recall+Rank): {'recall@k': 0.026433216608304153, 'ndcg@k': 0.011951752312429769} K = 10

Version 1.2:
User side: added favorite_genres (top-3 genre affinities) and a temporal preference signal (user_avg_release_year buckets into prefers_classics/balanced/prefers_recent) so the tower captures longer-term taste and recency bias.
Recall pipeline: Introduced multiple recall streams—co-visitation, item-based collaborative filtering, and a popularity prior—and a generalized merge_candidate_lists helper that blends FAISS/two-tower with co-vis, item-CF, and popularity candidates (default 45/25/20/10 split).
FAISS index: upgraded to support ANN IVF search with fallback to exact when the catalog is small, ready for scale.
Ranking: continues to score the merged candidate list using the two-tower user/item embeddings; the extra recall streams expand the pool without changing the embedding backbone.
Result:
Hybrid Recall Eval (after hard-neg): {'recall@k': 0.23990995497748874, 'ndcg@k': 0.04662188754030811}, K = 100
Ranker Eval (Hybrid Recall+Rank): {'recall@k': 0.03028514257128564, 'ndcg@k': 0.013483822228789712}, K = 10

Version 1.3:
Recall: Two-tower embeddings are L2-normalized, add temperature in InBatchSoftmaxLoss, default as 0.1. RecallDataset add optional easy-negatvie sampler that mixes random catalog items with hard negatives, plus cached per-user masks to keep sampling overhead low. 
Hybrid Recall Eval (after hard-neg): {'recall@k': 0.2625412706353177, 'ndcg@k': 0.05497410929710433}, K = 100
Currently without easy and harder neg, but change the two tower model: {'recall@k': 0.25002501250625314, 'ndcg@k': 0.048952612925014016}
Hybrid search without two-tower: {'recall@k': 0.2714457228614307, 'ndcg@k': 0.06350793298392508}
Hybrid search with two-tower: {'recall@k': 0.2744072036018009, 'ndcg@k': 0.059135587778206496}

Version 1.4:
Evaluation: Remove the user's training interactions from direct FAISS search for all recall stream, previously all model have higher chances recommending user already watched movies.
Hybrid recall: Improve on the logic, instead of enforcing integer quotas for each stream, use a weighted score then re-sort to give final outputs.
Pure two-tower: {'recall@k': 0.35567783891945975, 'ndcg@k': 0.08762459726244512}
Pure itemCF: {'recall@k': 0.3207903951975988, 'ndcg@k': 0.08738528906010223}
Hybrid Recall Eval (after hard-neg): {'recall@k': 0.3860630315157579, 'ndcg@k': 0.09800878740993864}
Ranker Eval (Hybrid Recall+Rank): {'recall@k': 0.04414207103551776, 'ndcg@k': 0.020137062885961527}

Version 1.5:
Recall: two-tower model remove the hard negative finetuning, on the validation set it hurts the result.
Rank: Add DeepFM and DCN as baseline to replace the old rank_mlp.
Feature engineering: include movie item title embedding using pretrain model: sentence-transformers/all-MiniLM-L6-v2, shrink from 384d to 64d for two-tower usage.
Restructure the code for readability.
Result:
Hybrid Recall Eval (after warm-up): {'recall@k': 0.40760380190095047, 'ndcg@k': 0.10393681505170957}
Using DeepFM: Ranker Eval (Hybrid Recall+Rank): {'recall@k': 0.05467733866933467, 'ndcg@k': 0.0254182768799728}
Using DCN:    Ranker Eval (Hybrid Recall+Rank): {'recall@k': 0.06407203601800901, 'ndcg@k': 0.03007650543344327}