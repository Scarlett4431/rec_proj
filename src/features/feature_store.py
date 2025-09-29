import pandas as pd
import torch


class FeatureStore:
    """
    Generalized feature store:
      - Numeric features (normalized float32)
      - Categorical features (embedding lookup via vocab index, long)
      - Bucketized numeric features (as categorical bins, long)
    """

    def __init__(self, df, id_col,
                 numeric_cols=None,
                 cat_cols=None,
                 multi_cat_cols=None,
                 bucket_cols=None,
                 bucket_bins=10):

        self.id_col = id_col
        self.numeric_cols = numeric_cols or []
        self.cat_cols = cat_cols or []
        self.multi_cat_cols = multi_cat_cols or []
        self.bucket_cols = bucket_cols or []
        self.bucket_bins = bucket_bins

        # Ensure integer IDs
        df = df.copy()
        df[id_col] = df[id_col].astype(int)

        # ---- Normalize numeric ----
        self.scalers = {}
        self.numeric_df = pd.DataFrame(index=df[id_col])
        for col in self.numeric_cols:
            mean = df[col].mean()
            std = df[col].std() + 1e-6
            self.scalers[col] = (mean, std)
            self.numeric_df[col] = ((df[col] - mean) / std).fillna(0.0)

        # ---- Bucketize numeric ----
        self.bucket_df = pd.DataFrame(index=df[id_col])
        self.bucket_vocab_sizes = {}
        for col in self.bucket_cols:
            self.bucket_df[col] = pd.cut(df[col], bins=bucket_bins, labels=False).fillna(0).astype(int)
            self.bucket_vocab_sizes[col] = bucket_bins

        # ---- Categorical ----
        self.cat_df = pd.DataFrame(index=df[id_col])
        self.cat_vocabs = {}
        for col in self.cat_cols:
            unique = df[col].dropna().unique().tolist()
            vocab = {val: idx + 1 for idx, val in enumerate(unique)}  # +1 for padding
            self.cat_vocabs[col] = vocab
            self.cat_df[col] = df[col].map(vocab).fillna(0).astype(int)

        # ---- Multi-categorical (lists of tokens) ----
        self.multi_cat_vocabs = {}
        self.multi_cat_data = {}
        self.multi_cat_max_lens = {}
        for col in self.multi_cat_cols:
            tokens_series = df[col].apply(
                lambda x: x if isinstance(x, (list, tuple)) else ([] if pd.isna(x) else [x])
            )
            unique_tokens = sorted({token for tokens in tokens_series for token in tokens})
            vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}  # 0 reserved for padding
            self.multi_cat_vocabs[col] = vocab

            id_map = {}
            max_len = 0
            for entity_id, tokens in zip(df[id_col], tokens_series):
                ids = [vocab[token] for token in tokens if token in vocab]
                id_map[int(entity_id)] = torch.tensor(ids, dtype=torch.long)
                if len(ids) > max_len:
                    max_len = len(ids)
            self.multi_cat_data[col] = id_map
            self.multi_cat_max_lens[col] = max_len

        # ---- Build lookup ----
        self.data = {}
        for _, row in df.iterrows():
            idx = int(row[id_col])

            numeric_feats = (
                torch.tensor(self.numeric_df.loc[idx].values, dtype=torch.float32)
                if self.numeric_cols else torch.zeros(0, dtype=torch.float32)
            )
            cat_feats = {
                col: torch.tensor(int(self.cat_df.loc[idx, col]), dtype=torch.long)
                for col in self.cat_cols
            }
            bucket_feats = {
                col: torch.tensor(int(self.bucket_df.loc[idx, col]), dtype=torch.long)
                for col in self.bucket_cols
            }

            multi_cat_feats = {}
            for col in self.multi_cat_cols:
                seq = self.multi_cat_data[col].get(idx, torch.zeros(0, dtype=torch.long))
                max_len = self.multi_cat_max_lens.get(col, 0)
                if max_len > 0:
                    seq = seq[:max_len]
                    if len(seq) < max_len:
                        seq = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
                multi_cat_feats[col] = seq

            self.data[idx] = {
                "numeric": numeric_feats,
                "categorical": cat_feats,
                "bucket": bucket_feats,
                "multi_categorical": multi_cat_feats,
            }

    def get(self, idx: int):
        """Return dict of feature groups for one entity (idx must be int)."""
        idx = int(idx)
        return self.data.get(idx, {
            "numeric": torch.zeros(len(self.numeric_cols), dtype=torch.float32),
            "categorical": {c: torch.tensor(0, dtype=torch.long) for c in self.cat_cols},
            "bucket": {c: torch.tensor(0, dtype=torch.long) for c in self.bucket_cols},
            "multi_categorical": {
                c: torch.zeros(self.multi_cat_max_lens.get(c, 0), dtype=torch.long)
                for c in self.multi_cat_cols
            },
        })

    def get_batch(self, ids, max_multi_lengths=None):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        max_multi_lengths = max_multi_lengths or {}

        numeric_list = [] if self.numeric_cols else None
        cat_lists = {col: [] for col in self.cat_cols}
        bucket_lists = {col: [] for col in self.bucket_cols}
        multi_lists = {col: [] for col in self.multi_cat_cols}

        for idx in ids:
            feats = self.get(idx)
            if numeric_list is not None:
                numeric_list.append(feats["numeric"])
            for col in self.cat_cols:
                cat_lists[col].append(feats["categorical"][col])
            for col in self.bucket_cols:
                bucket_lists[col].append(feats["bucket"][col])
            for col in self.multi_cat_cols:
                seq = feats["multi_categorical"][col]
                multi_lists[col].append(seq)

        result = {}
        if numeric_list is not None:
            result["numeric"] = torch.stack(numeric_list) if numeric_list else torch.zeros((0, len(self.numeric_cols)))
        else:
            result["numeric"] = torch.zeros((len(ids), 0)) if ids else torch.zeros((0, 0))

        result["categorical"] = {
            col: torch.stack(tensors) if tensors else torch.zeros((len(ids),), dtype=torch.long)
            for col, tensors in cat_lists.items()
        }
        result["bucket"] = {
            col: torch.stack(tensors) if tensors else torch.zeros((len(ids),), dtype=torch.long)
            for col, tensors in bucket_lists.items()
        }

        multi_cat = {}
        multi_lengths = {}
        for col, sequences in multi_lists.items():
            if not sequences:
                multi_cat[col] = torch.zeros((len(ids), 0), dtype=torch.long)
                multi_lengths[col] = torch.zeros(len(ids), dtype=torch.long)
                continue

            max_len = max_multi_lengths.get(col)
            if max_len is None:
                max_len = max((len(seq) for seq in sequences), default=0)
            padded = []
            lengths = []
            for seq in sequences:
                seq_len = len(seq)
                seq = seq[:max_len]
                lengths.append(min(seq_len, max_len))
                if len(seq) < max_len:
                    seq = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
                padded.append(seq)
            multi_cat[col] = torch.stack(padded)
            multi_lengths[col] = torch.tensor(lengths, dtype=torch.float32).unsqueeze(-1)

        result["multi_categorical"] = multi_cat
        result["multi_lengths"] = multi_lengths

        return result

    def get_tensor(self, idx: int):
        """
        Return concatenated tensor of all features (numeric + categorical + bucketized).
        - numeric kept as float32
        - categorical/bucket cast to float32 for concatenation
        """
        feats = self.get(idx)
        numeric = feats["numeric"]
        cat = [feats["categorical"][c].float() for c in self.cat_cols]
        bucket = [feats["bucket"][c].float() for c in self.bucket_cols]
        multi_vectors = []
        for col in self.multi_cat_cols:
            max_len = self.multi_cat_max_lens.get(col, 0)
            if max_len == 0:
                continue
            seq = feats["multi_categorical"].get(col)
            if seq is None or seq.numel() == 0:
                seq = torch.zeros(max_len, dtype=torch.long)
            else:
                seq = seq[:max_len]
                if seq.numel() < max_len:
                    seq = torch.cat([seq, torch.zeros(max_len - seq.numel(), dtype=torch.long)])
            multi_vectors.append(seq.float().view(-1))

        if cat:
            cat = torch.stack(cat)
        else:
            cat = torch.zeros(0, dtype=torch.float32)

        if bucket:
            bucket = torch.stack(bucket)
        else:
            bucket = torch.zeros(0, dtype=torch.float32)

        if multi_vectors:
            multi = torch.cat(multi_vectors, dim=0)
        else:
            multi = torch.zeros(0, dtype=torch.float32)

        parts = [numeric, cat, bucket, multi]
        parts = [p.flatten() for p in parts if p.numel() > 0]
        if not parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    def to_numeric_matrix(self, max_id: int):
        """[N, F] numeric features only."""
        mat = torch.zeros((max_id + 1, len(self.numeric_cols)), dtype=torch.float32)
        for idx, feats in self.data.items():
            idx = int(idx)
            if idx <= max_id and len(feats["numeric"]) > 0:
                mat[idx] = feats["numeric"]
        return mat

    def to_matrix(self, max_id: int):
        """
        [N, F] tensor of all concatenated features (numeric + categorical + bucketized).
        """
        total_dim = self.total_dim
        mat = torch.zeros((max_id + 1, total_dim), dtype=torch.float32)
        for idx in range(max_id + 1):
            mat[idx] = self.get_tensor(idx)
        return mat

    # ------------------- Properties -------------------
    @property
    def numeric_dim(self):
        return len(self.numeric_cols)

    @property
    def cat_dims(self):
        """Return vocab sizes for each categorical col (needed for nn.Embedding)."""
        return {col: len(vocab) + 1 for col, vocab in self.cat_vocabs.items()}

    @property
    def bucket_dims(self):
        return {col: self.bucket_bins for col in self.bucket_cols}

    @property
    def total_dim(self):
        multi_dim = sum(self.multi_cat_max_lens.values()) if self.multi_cat_max_lens else 0
        return len(self.numeric_cols) + len(self.cat_cols) + len(self.bucket_cols) + multi_dim

    @property
    def multi_cat_dims(self):
        return {col: len(vocab) + 1 for col, vocab in self.multi_cat_vocabs.items()}
