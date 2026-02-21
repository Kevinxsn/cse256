# add all  your Encoder and Decoder code here

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Precompute positional encodings for maximum sequence length
        self.encoding = torch.zeros(max_len, d_model)
        
        # Calculate the positional encodings using sine and cosine functions
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices use sine
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices use cosine
        
        # Add a batch dimension for broadcasting
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding.requires_grad = False

    def forward(self, x):
        # Move encoding to the same device as input and add it to the input
        encoding = self.encoding.to(x.device)
        return x + encoding[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (B, Lq, d_model) -> (B, L, h, d_k)
        K = self.key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # tanspose: (B, h, Lk, d_k) → (B, h, d_k, Lk)
        # matual: (B, h, Lq, d_k) @ (B, h, d_k, Lk) = (B, h, Lq, Lk)
        
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim = -1)
        output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, self.h*self.d_k)
        
        # attention: (B, h, Lq, Lk)
        # V: (B, h, Lk, d_k)
        # result:   (B, h, Lq, d_k)
        
        return self.fc(output), attention
    

class FeedForward(nn.Module):
    """Implements position-wise feed-forward network with dropout and ReLU."""
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Apply ReLU activation after first linear layer
        x = self.dropout(x)  # Apply dropout
        return self.linear2(x)  # Final transformation back to original dimension

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)  # Multi-head attention
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # Feed-forward layer
        self.ln1 = nn.LayerNorm(d_model)  # Layer normalization after attention
        self.ln2 = nn.LayerNorm(d_model)  # Layer normalization after feed-forward
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention
        self.dropout2 = nn.Dropout(dropout)  # Dropout after feed-forward

    def forward(self, src, src_mask):
        src2, attn_map = self.attention(src, src, src, src_mask)  # Multi-head attention with mask
        src = src + self.dropout1(src2)  # Add residual connection
        src = self.ln1(src)  # Normalize
        src2 = self.feed_forward(src)  # Feed-forward transformation
        src = src + self.dropout2(src2)  # Add residual connection
        src = self.ln2(src)  # Normalize again
        return src, attn_map  # Return both output and attention map


class TransformerEncoder(nn.Module):
    """Defines the entire encoder with embedding, positional encoding, and multiple encoder layers."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)  # Token embedding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)  # Positional encoding
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src = self.encoder_embedding(src)  # Apply token embedding
        src = self.positional_encoding(src)  # Add positional encoding
        src = self.dropout(src)  # Apply dropout
        
        attentions = []
        for layer in self.encoder_layers:
            src, attn_map = layer(src, src_mask)  # Apply each encoder layer
            attentions.append(attn_map)  # Collect attention maps for visualization
        return src, attentions




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


class MultiHeadAttentionDecoder(nn.Module):
    """Vectorized Multi-head attention for the decoder with causal masking."""
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        self.h = n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Project and split heads: (B, L, D) -> (B, L, h, d_k) -> (B, h, L, d_k)
        Q = self.query(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        mask = self.tril[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Concatenate heads and project back: (B, h, L, d_k) -> (B, L, D)
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.fc(out), attn_weights

class Block(nn.Module):
    """Simplified Decoder block using the vectorized attention."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttentionDecoder(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForwardDecoder(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention + Residual + LayerNorm
        attn_out, attn_map = self.sa(x)
        x = self.ln1(x + attn_out)
        
        # FeedForward + Residual + LayerNorm
        x = self.ln2(x + self.ffwd(x))
        return x, attn_map
    
    
class TransformerDecoder(nn.Module):
    """Decoder-only Transformer Language Model with stacked blocks."""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  #(vocab_size, D)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)  
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Language model head for predicting tokens

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # Token embeddings (B,T,D), D = n_emb
        # Standardize position embeddings to use the same device as the input
        pos_emb = self.pos_embedding_table(torch.arange(T, device=idx.device))# Positional embeddings (T,D)
        x = tok_emb + pos_emb
        
        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map) # Returns (B, H, L, L) per layer

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return loss, attn_maps