import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseMultiHeadAttentionDecoder(nn.Module):
    def __init__(self, d_model, n_heads, block_size, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.window_size = window_size # Number of past tokens to attend to

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 1. Standard Causal Mask (Lower Triangular)
        tril = torch.tril(torch.ones(block_size, block_size))
        
        # 2. Sliding Window Mask (Banded Matrix)
        # Keep only values within 'window_size' of the diagonal
        band = torch.triu(torch.ones(block_size, block_size), diagonal=-window_size + 1)
        
        # 3. Combined Sparse Causal Mask
        # A token at index 'i' only attends to indices [i - window_size + 1, i]
        sparse_mask = tril * band
        self.register_buffer('sparse_mask', sparse_mask)

    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.query(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        K = self.key(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        V = self.value(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply the precomputed Sparse Causal Mask
        mask = self.sparse_mask[:L, :L]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.fc(out), attn_weights


class FeedForwardDecoder(nn.Module):
    """Feed-forward layer in the decoder with a ReLU activation."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 100),  # First layer with 100 units
            nn.ReLU(),
            nn.Linear(100, n_embd),  # Output layer matching input dimension
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SparseBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, window_size, dropout):
        super().__init__()
        self.sa = SparseMultiHeadAttentionDecoder(n_embd, n_head, block_size, window_size, dropout)
        self.ffwd = FeedForwardDecoder(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_out, attn_map = self.sa(x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffwd(x))
        return x, attn_map

class SparseDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, window_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        # Use SparseBlocks instead of standard Blocks
        self.blocks = nn.ModuleList([
            SparseBlock(n_embd, n_head, block_size, window_size, dropout) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.pos_embedding_table(torch.arange(T, device=idx.device))
        
        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return loss, attn_maps