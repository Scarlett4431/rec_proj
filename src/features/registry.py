USER_FEATURES = [
    "user_total_ratings",
    "user_avg_rating",
    "user_recency_days"
]

ITEM_FEATURES = [
    "item_total_ratings",
    "item_avg_rating",
    "genres_embedding"  # optional: one-hot or pretrained
]

CONTEXT_FEATURES = [
    "days_since_interaction",
]