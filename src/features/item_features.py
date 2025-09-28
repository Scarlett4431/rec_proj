import numpy as np
import pandas as pd


def build_item_features(ratings, movies):
    item_stats = ratings.groupby("item_idx").agg(
        item_total_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index()

    movies = movies.copy()

    # Extract release year from title (e.g., "Toy Story (1995)")
    movies["item_release_year"] = (
        movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    )

    # Build multi-hot genre indicators
    genre_lists = movies["genres"].str.split("|")
    unique_genres = sorted({g for genres in genre_lists for g in (genres or []) if g and g != "(no genres listed)"})
    for genre in unique_genres:
        col_name = f"genre_{genre.replace(' ', '_').replace('-', '_')}"
        movies[col_name] = genre_lists.apply(lambda gl: 1.0 if gl and genre in gl else 0.0)

    # Merge with rating stats
    item_df = movies.merge(item_stats, on="item_idx", how="left")

    # Fill missing numeric values
    numeric_cols = ["item_total_ratings", "item_avg_rating", "item_release_year"] + [
        col for col in item_df.columns if col.startswith("genre_")
    ]
    for col in numeric_cols:
        if col in item_df.columns:
            item_df[col] = item_df[col].fillna(item_df[col].mean() if col != "item_total_ratings" else 0.0)

    return item_df
