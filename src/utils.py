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