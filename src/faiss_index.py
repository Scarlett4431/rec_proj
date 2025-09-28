import faiss
import numpy as np


class FaissIndex:
    def __init__(self, item_emb, use_ann=False, nlist=None, nprobe=32):
        """Build a FAISS index (IVF when possible, otherwise exact).

        Args:
            item_emb: torch.Tensor or np.ndarray with shape [num_items, dim]
            use_ann: enable approximate nearest neighbour search when feasible
            nlist: number of IVF clusters; defaults to sqrt(num_items)
            nprobe: number of clusters to probe at query time
        """
        if hasattr(item_emb, "cpu"):
            item_np = item_emb.cpu().numpy().astype(np.float32)
        else:
            item_np = np.asarray(item_emb, dtype=np.float32)

        if item_np.ndim != 2:
            raise ValueError("item_emb must be 2D: [num_items, dim]")

        num_items, dim = item_np.shape
        self.item_emb = item_np

        # Decide between exact and ANN search.
        self.use_ann = use_ann and num_items >= 1000
        if self.use_ann:
            nlist = nlist or max(1, int(np.sqrt(num_items)))
            nlist = min(nlist, num_items)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(item_np)
            index.add(item_np)
            index.nprobe = min(nprobe, nlist)
            self.index = index
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(item_np)
            self.index = index

    def search(self, user_vec, k=10):
        """Return top-k (scores, indices) for a user embedding."""
        if hasattr(user_vec, "cpu"):
            user_np = user_vec.cpu().numpy().astype(np.float32).reshape(1, -1)
        else:
            user_np = np.asarray(user_vec, dtype=np.float32).reshape(1, -1)
        scores, idxs = self.index.search(user_np, k)
        return scores[0], idxs[0]
