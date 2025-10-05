import torch
import torch.nn.functional as F
import torch.nn as nn

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Args:
        pos_scores: shape [B, 1] or [B]
        neg_scores: shape [B, K]
    Returns:
        scalar loss
    """
    if pos_scores.dim() == 1:
        pos_scores = pos_scores.unsqueeze(-1)
    diff = pos_scores - neg_scores  # [B, K]
    return -weight * torch.mean(F.logsigmoid(diff))


class InBatchSoftmaxLoss(nn.Module):
    """
    InfoNCE-style loss for in-batch negatives.
    Each user's positive item: all other items in batch are negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        # user_emb: [B, d], item_emb: [B, d]
        logits = torch.matmul(user_emb, item_emb.T)  # [B, B]
        logits = logits / max(self.temperature, 1e-6)
        labels = torch.arange(len(user_emb), device=user_emb.device)
        return self.ce(logits, labels)