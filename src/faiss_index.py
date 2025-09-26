import faiss
import numpy as np

class FaissIndex:
    def __init__(self, item_emb):
        """
        item_emb: torch.Tensor [num_items, dim]
        """
        self.item_emb = item_emb.cpu().numpy().astype(np.float32)
        d = self.item_emb.shape[1]
        self.index = faiss.IndexFlatIP(d)  # inner product search
        self.index.add(self.item_emb)

    def search(self, user_vec, k=10):
        """
        user_vec: torch.Tensor [d]
        Returns: top-k (scores, indices)
        """
        user_np = user_vec.cpu().numpy().astype(np.float32).reshape(1, -1)
        scores, idxs = self.index.search(user_np, k)
        return scores[0], idxs[0]