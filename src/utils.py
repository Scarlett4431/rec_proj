from collections import defaultdict

import pandas as pd
import numpy as np


def remap_ids(ratings, movies):
    # Factorize users
    ratings["user_idx"], user_index = pd.factorize(ratings["userId"])
    # Factorize items
    ratings["item_idx"], item_index = pd.factorize(ratings["movieId"])

    # Apply same mapping to movies table
    movies = movies.copy()
    movies = movies.set_index("movieId").loc[item_index].reset_index(drop=True)
    movies["item_idx"] = np.arange(len(item_index))

    user2id = dict(enumerate(user_index))   # idx -> raw userId
    item2id = dict(enumerate(item_index))   # idx -> raw movieId

    return ratings, movies, user2id, item2id


def user_stratified_split(ratings, test_frac=0.1, random_state=42, min_test_items=1):
    """Split ratings so every user keeps interactions in train and (optionally) test.

    Ensures no user appears exclusively in the test set, avoiding cold-start users
    during evaluation. Users with too few interactions remain entirely in train.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")

    train_parts = []
    test_parts = []

    for user_id, group in ratings.groupby("user_idx", sort=False):
        group_size = len(group)
        # Need at least (min_test_items + 1) interactions to contribute to test set
        min_required = max(min_test_items + 1, 2)
        if group_size < min_required:
            train_parts.append(group)
            continue

        test_size = max(min_test_items, int(round(group_size * test_frac)))
        if test_size >= group_size:
            test_size = group_size - 1
        if test_size <= 0:
            train_parts.append(group)
            continue

        sampled = group.sample(n=test_size, random_state=random_state + user_id)
        remaining = group.drop(sampled.index)

        train_parts.append(remaining)
        test_parts.append(sampled)

    train_df = pd.concat(train_parts).sort_index().reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_index().reset_index(drop=True) if test_parts else pd.DataFrame(columns=ratings.columns)

    return train_df, test_df


def build_user_item_map(df: pd.DataFrame, user_col: str = "user_idx", item_col: str = "item_idx"):
    """Return dict mapping each user to the set of interacted item indices."""

    interactions = defaultdict(set)
    if df.empty:
        return interactions

    users = df[user_col].to_numpy()
    items = df[item_col].to_numpy()
    for u, i in zip(users, items):
        interactions[int(u)].add(int(i))

    return interactions
