import pandas as pd

def build_user_features(ratings):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    max_date = ratings['timestamp'].max()

    user_df = ratings.groupby("user_idx").agg(
        user_total_ratings=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        last_rating=("timestamp", "max")
    ).reset_index()
    user_df["user_recency_days"] = (max_date - user_df["last_rating"]).dt.days
    return user_df.drop(columns=["last_rating"])
