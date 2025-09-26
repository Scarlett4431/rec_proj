import pandas as pd

def load_movielens_1m(data_dir="data/ml-1m"):
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )
    return ratings, movies