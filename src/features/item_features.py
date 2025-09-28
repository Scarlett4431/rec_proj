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

    return item_df
