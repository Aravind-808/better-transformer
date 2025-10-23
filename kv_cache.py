import torch

class KVCaching:
    """
    KV cache for autoregressive multi-head attention.
    Stores keys and values and allows incremental updates.
    """
    def __init__(self, batch, heads, max_seq_len, d_k, device=None):
        self.batch = batch
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.device = device

        # Initialize empty cache
        self.k_cache = torch.zeros(batch, heads, max_seq_len, d_k, device=device)
        self.v_cache = torch.zeros(batch, heads, max_seq_len, d_k, device=device)
        self.cache_idx = 0

    def update_cache(self, K_new, V_new):
        """
        Add new keys/values to the cache.
        K_new, V_new: shape (B, H, T, d_k)
        """
        T = K_new.shape[2]  # number of new tokens

        self.k_cache[:, :, self.cache_idx:self.cache_idx+T, :] = K_new
        self.v_cache[:, :, self.cache_idx:self.cache_idx+T, :] = V_new

        self.cache_idx = min(self.cache_idx + T, self.max_seq_len)

    def get_cache(self):
        # return cached k and v upto the current index
        return self.k_cache[:, :, :self.cache_idx, :], self.v_cache[:, :, :self.cache_idx, :]

    # def reset_cache(self):
    #     self.k_cache.zero_()
    #     self.v_cache.zero_()
    #     self.cache_idx = 0