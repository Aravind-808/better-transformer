import torch
import torch.nn as nn
import math
from rope import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    '''
    modified multihead attention block with rope configuration
    '''
    def __init__(self, d_model, heads, max_seq_len):
        super().__init__()
        assert d_model % heads ==0 # number of heads MUST be divisible by dimensions

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model//heads

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)  

        self.rope = RotaryPositionalEmbedding(self.d_k, self.max_seq_len)

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
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d)

    # main mechanism that is done during inference
    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        Q, K = self.rope(Q, K)

        attention_output = self.scaled_dotproduct_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output
