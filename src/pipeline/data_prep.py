from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.data_loader import load_movielens_1m
from src.utils import remap_ids, user_stratified_split, build_user_item_map


@dataclass
class PreparedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    train_all: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]
    val_users: List[int]
    user_consumed: Dict[int, set]
    num_users: int
    num_items: int
    user2id: Dict[int, int]
    item2id: Dict[int, int]


def load_and_prepare_data(data_dir: str = "data/ml-1m") -> PreparedData:
    ratings, movies = load_movielens_1m(data_dir)
    ratings, movies, user2id, item2id = remap_ids(ratings, movies)

    max_user_id = int(ratings["user_idx"].max())
    max_item_id = int(ratings["item_idx"].max())
    num_users = max_user_id + 1
    num_items = max_item_id + 1

    train_all, test = user_stratified_split(ratings, test_frac=0.1, random_state=42)
    train, val = user_stratified_split(train_all, test_frac=0.1, random_state=7)

    test_pairs = list(zip(test.user_idx.values.tolist(), test.item_idx.values.tolist()))
    val_pairs = list(zip(val.user_idx.values.tolist(), val.item_idx.values.tolist()))
    val_users = sorted(val.user_idx.unique().tolist())

    user_consumed = build_user_item_map(train_all)
    val_holdout = build_user_item_map(val)
    test_holdout = build_user_item_map(test)

    for user, items in val_holdout.items():
        if user in user_consumed:
            user_consumed[user].difference_update(items)
    for user, items in test_holdout.items():
        if user in user_consumed:
            user_consumed[user].difference_update(items)

    return PreparedData(
        ratings=ratings,
        movies=movies,
        train_all=train_all,
        train=train,
        val=val,
        test=test,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        val_users=val_users,
        user_consumed=user_consumed,
        num_users=num_users,
        num_items=num_items,
        user2id=user2id,
        item2id=item2id,
    )
