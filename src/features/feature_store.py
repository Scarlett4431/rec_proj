import torch

class FeatureStore:
    """
    Utility to store numeric features for entities (users, items).
    Provides dict lookup + default vector + aligned full matrix export.
    """

    def __init__(self, df, id_col):
        """
        df: DataFrame with [id_col, feat1, feat2, ...]
        id_col: "userId" or "movieId"
        """
        if df is None or df.empty:
            self.feat_dict = {}
            self.default = torch.zeros(0, dtype=torch.float32)
            self.dim = 0
            return

        numeric = df.set_index(id_col).select_dtypes(include="number").fillna(0.0)

        self.feat_dict = {
            idx: torch.tensor(row.values, dtype=torch.float32)
            for idx, row in numeric.iterrows()
        }
        self.default = torch.zeros(numeric.shape[1], dtype=torch.float32)
        self.dim = numeric.shape[1]

    def get(self, idx):
        """Return feature vector for id (or default if missing)."""
        return self.feat_dict.get(idx, self.default)

    def to_matrix(self, max_id):
        """Build aligned dense matrix [max_id+1, dim]."""
        mat = torch.zeros((max_id + 1, self.dim), dtype=torch.float32)
        for idx, feats in self.feat_dict.items():
            if idx <= max_id:
                mat[idx] = feats
        return mat