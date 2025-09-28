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
Recall pipeline: Introduced two additional recall streams—co-visitation and item-based collaborative filtering—and a generalized merge_candidate_lists helper that blends FAISS/two-tower, co-vis, and item-CF candidates (50/30/20 mix by default).
FAISS index: upgraded to support ANN IVF search with fallback to exact when the catalog is small, ready for scale.
Result:
FAISS Recall Eval (after hard-neg): {'recall@k': 0.14838419209604803, 'ndcg@k': 0.03176630032187134}
