def build_item_features(ratings, movies):
    item_stats = ratings.groupby("item_idx").agg(
        item_total_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index()
    item_df = movies.merge(item_stats, on="item_idx", how="left")
    return item_df