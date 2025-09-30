import torch
import torch.nn as nn


class SASRecRanker(nn.Module):
    """SASRec-style ranker operating on item embedding sequences."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        max_history: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.max_history = max_history
        self.requires_history = True

        self.position_emb = nn.Embedding(max_history, item_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=item_dim,
            nhead=num_heads,
            dim_feedforward=item_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.user_project = nn.Linear(user_dim, item_dim)

        dnn_input_dim = item_dim * 3 + user_feat_dim + item_feat_dim  # hist_rep, target_emb, user_proj, side feats
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        u_feats: torch.Tensor = None,
        i_feats: torch.Tensor = None,
        hist_emb: torch.Tensor = None,
        hist_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if hist_emb is None or hist_mask is None:
            raise ValueError("SASRecRanker requires history embeddings and mask")

        batch_size, max_len, _ = hist_emb.size()
        device = hist_emb.device
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_emb(positions)
        transformer_input = hist_emb + pos_emb
        src_padding_mask = ~hist_mask
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=src_padding_mask)

        lengths = hist_mask.sum(dim=1).clamp(min=1)
        gather_idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.item_dim)
        last_hidden = transformer_output.gather(1, gather_idx).squeeze(1)
        last_hidden = torch.where(lengths.unsqueeze(-1) > 0, last_hidden, torch.zeros_like(last_hidden))

        target_emb = i_emb
        user_rep = self.user_project(u_emb)
        parts = [last_hidden, target_emb, user_rep]
        if u_feats is not None and u_feats.numel() > 0:
            parts.append(u_feats)
        if i_feats is not None and i_feats.numel() > 0:
            parts.append(i_feats)

        dnn_input = torch.cat(parts, dim=-1)
        logits = self.dnn(dnn_input)
        return torch.sigmoid(logits).squeeze(-1)
