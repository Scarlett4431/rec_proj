import pandas as pd
def build_context_features(ratings):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings["days_since_interaction"] = (
        ratings['timestamp'].max() - ratings['timestamp']
    ).dt.days
    return ratings[["user_idx", "item_idx", "days_since_interaction"]]