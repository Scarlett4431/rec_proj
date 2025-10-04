import re

import numpy as np
import pandas as pd


def build_item_features(ratings, movies):
    item_stats = ratings.groupby("item_idx").agg(
        item_total_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean"),
        item_rating_variance=("rating", "var")
    ).reset_index()
    item_stats["item_rating_variance"] = item_stats["item_rating_variance"].fillna(0.0)

    movies = movies.copy()

    # Extract release year from title (e.g., "Toy Story (1995)")
    movies["item_release_year"] = (
        movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    )
    movies["item_genres"] = (
        movies["genres"].fillna("")
        .str.split("|")
        .apply(lambda gl: [g for g in gl if g and g != "(no genres listed)"])
    )

    # Merge with rating stats
    item_df = movies.merge(item_stats, on="item_idx", how="left")

    # Fill missing numeric values
    numeric_cols = ["item_total_ratings", "item_avg_rating", "item_release_year"]
    for col in numeric_cols:
        if col in item_df.columns:
            item_df[col] = item_df[col].fillna(item_df[col].mean() if col != "item_total_ratings" else 0.0)

    # Popularity & recency signals
    interaction_counts = ratings["item_idx"].value_counts().to_dict()
    item_df["item_interaction_count"] = item_df["item_idx"].map(interaction_counts).fillna(0).astype(float)
    if "timestamp" in ratings.columns:
        ratings_ts = ratings[["item_idx", "timestamp"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(ratings_ts["timestamp"]):
            ratings_ts["timestamp"] = pd.to_datetime(ratings_ts["timestamp"], unit="s")
        latest_ts = ratings_ts["timestamp"].max()
        last_touch = ratings_ts.groupby("item_idx")["timestamp"].max()
        recency_days = (latest_ts - last_touch).dt.days.astype(float)
        item_df["item_recency_days"] = item_df["item_idx"].map(recency_days).fillna(recency_days.mean() if len(recency_days) else 0.0)
    else:
        item_df["item_recency_days"] = float(0.0)

    total_items = max(len(item_df), 1)
    ranks = (
        item_df["item_interaction_count"]
        .rank(method="dense", ascending=False)
        .fillna(len(item_df))
    )
    item_df["item_popularity_rank"] = (ranks - 1) / max(total_items - 1, 1)
    item_df["item_interaction_log"] = np.log1p(item_df["item_interaction_count"])

    # Genre richness
    item_df["item_genre_count"] = item_df["item_genres"].apply(lambda g: len(g) if isinstance(g, list) else 0)

    return item_df
