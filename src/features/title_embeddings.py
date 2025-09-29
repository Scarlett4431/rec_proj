import math
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class TitleEmbeddingEncoder:
    """Utility to encode item titles using a frozen pretrained language model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 256,
        normalize: bool = True,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a sequence of texts and return a tensor [N, D].

        Args:
            texts: iterable of strings
            batch_size: encoding batch size
            normalize: if True, L2-normalize each embedding
            max_length: optional max token length
        """
        texts = list(texts)
        if not texts:
            return torch.zeros((0, self.model.config.hidden_size), dtype=torch.float32)

        outputs: List[torch.Tensor] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_out = self.model(**tokens)
                hidden = model_out.last_hidden_state  # [B, L, D]
                mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-6)
                sentence_emb = summed / counts
                if normalize:
                    sentence_emb = torch.nn.functional.normalize(sentence_emb, p=2, dim=1)
            outputs.append(sentence_emb.cpu())

        return torch.cat(outputs, dim=0)


def load_or_compute_title_embeddings(
    titles: Iterable[str],
    cache_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load cached embeddings if available; otherwise compute and save to cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return torch.load(cache_path)

    encoder = TitleEmbeddingEncoder(model_name=model_name, device=device)
    embeddings = encoder.encode(titles, batch_size=batch_size)
    torch.save(embeddings, cache_path)
    return embeddings
