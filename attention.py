import torch
import torch.nn as nn
import math
from rope import RotaryPositionalEmbedding
from kv_cache import KVCaching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiHeadAttention(nn.Module):
    '''
    modified multihead attention block with 
    - rope configuration
    - KV caching
    '''
    def __init__(self, d_model, heads, max_seq_len):
        super().__init__()
        assert d_model % heads ==0 # number of heads MUST be divisible by dimensions

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model//heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)  

        self.rope = RotaryPositionalEmbedding(self.d_k, self.max_seq_len)
        self.kv_cache = None

    def scaled_dotproduct_attention(self, Q, K, V, mask = None):
        # attention(q,k,v) = softmax((q*k_transpose)/root(dimensionality)) * v

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None: # apply the mask (optional)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probabilities = torch.softmax(attention_scores, dim = -1)

        output = torch.matmul(attention_probabilities, V)

        return output

    # split the heads induvidually
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)
    
    # combine them back
    def combine_heads(self, x):
        batch_size, num, seq_length,  d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    # main mechanism that is done during inference
    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # initialize k_cache and v_cache
        batch, head, token_size, _ = K.shape # V's shape is also same

        if self.kv_cache is None:
            self.kv_cache = KVCaching(
                batch=batch,
                heads= head,
                max_seq_len=self.max_seq_len,
                d_k= self.d_k,
                device = device
                )
        # keep track of the current index in seq
        start_pos = self.kv_cache.cache_idx

        # apply rope wit current pos offset
        Q, K = self.rope(Q, K, pos_offset = start_pos)

        # update caches in place
        self.kv_cache.update_cache(K, V)
        # get new cached values
        K, V = self.kv_cache.get_cache()

        # multihead attention stuff
        attention_output = self.scaled_dotproduct_attention(Q,K,V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        
        return output