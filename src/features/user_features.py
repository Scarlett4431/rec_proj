import pandas as pd


def build_user_features(ratings):
    ratings = ratings.copy()
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    max_date = ratings["timestamp"].max()

    user_basic = ratings.groupby("user_idx").agg(
        user_total_ratings=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        last_rating=("timestamp", "max")
    ).reset_index()
    user_basic["user_recency_days"] = (max_date - user_basic["last_rating"]).dt.days
    user_basic = user_basic.drop(columns=["last_rating"])

    ratings_sorted = ratings.sort_values(["user_idx", "timestamp"], ascending=[True, False])
    watched_sequences = ratings_sorted.groupby("user_idx")
    watched_map = {
        user: group["item_idx"].tolist()
        for user, group in watched_sequences
    }

    user_basic["watched_items"] = user_basic["user_idx"].map(lambda u: watched_map.get(u, []))
    return user_basic
