import pandas as pd


def build_user_features(ratings, movies, top_k_genres=3, temporal_margin=5.0):
    """Create per-user aggregates including recent activity, history, and preference signals."""
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

    # Build per-user genre preference list (top-k by interaction count, tie-broken by rating sum).
    movies = movies.copy()
    movies["item_genres"] = (
        movies["genres"].fillna("")
        .str.split("|")
        .apply(lambda gl: [g for g in gl if g and g != "(no genres listed)"])
    )
    if "item_release_year" not in movies.columns:
        movies["item_release_year"] = (
            movies["title"].str.extract(r"\((\d{4})\)").astype(float)
        )
    movies["item_release_year"] = movies["item_release_year"].fillna(movies["item_release_year"].mean())

    ratings_with_meta = ratings.merge(
        movies[["item_idx", "item_genres", "item_release_year"]],
        on="item_idx",
        how="left",
    )

    exploded = ratings_with_meta.explode("item_genres")
    exploded = exploded.dropna(subset=["item_genres"])

    if not exploded.empty:
        genre_pref = exploded.groupby(["user_idx", "item_genres"]).agg(
            genre_count=("item_genres", "size"),
            rating_sum=("rating", "sum"),
        ).reset_index()

        genre_pref = genre_pref.sort_values(
            ["user_idx", "genre_count", "rating_sum"],
            ascending=[True, False, False],
        )
        genre_pref_map = {
            user: group["item_genres"].head(top_k_genres).tolist()
            for user, group in genre_pref.groupby("user_idx")
        }
    else:
        genre_pref_map = {}

    user_basic["watched_items"] = user_basic["user_idx"].map(lambda u: watched_map.get(u, []))
    user_basic["favorite_genres"] = user_basic["user_idx"].map(lambda u: genre_pref_map.get(u, []))

    # Temporal preference: weighted average release year + categorical preference bucket.
    release_years = ratings_with_meta.dropna(subset=["item_release_year"])
    global_year_mean = release_years["item_release_year"].mean()

    def _weighted_year(group):
        years = group["item_release_year"].to_numpy()
        ratings_arr = group["rating"].to_numpy()
        weight_sum = ratings_arr.sum()
        if weight_sum <= 0:
            return float(years.mean()) if len(years) else float(global_year_mean)
        return float((years * ratings_arr).sum() / weight_sum)

    if not release_years.empty:
        user_year_pref = release_years.groupby("user_idx").apply(_weighted_year).reset_index(name="user_avg_release_year")
    else:
        user_year_pref = pd.DataFrame({"user_idx": [], "user_avg_release_year": []})

    user_basic = user_basic.merge(user_year_pref, on="user_idx", how="left")
    fallback_year = global_year_mean if pd.notna(global_year_mean) else 0.0
    user_basic["user_avg_release_year"] = user_basic["user_avg_release_year"].fillna(fallback_year)

    mean_year = global_year_mean if pd.notna(global_year_mean) else user_basic["user_avg_release_year"].mean()

    def _temporal_bucket(avg_year):
        if pd.isna(avg_year) or pd.isna(mean_year):
            return "balanced"
        if avg_year <= mean_year - temporal_margin:
            return "prefers_classics"
        if avg_year >= mean_year + temporal_margin:
            return "prefers_recent"
        return "balanced"

    user_basic["temporal_preference"] = user_basic["user_avg_release_year"].apply(_temporal_bucket)

    return user_basic
