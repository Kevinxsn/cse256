# DANModels.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples  # same as BOWModels.py uses


class SentimentDatasetDAN(Dataset):
    """
    Returns:
      x: LongTensor [max_len] of word indices (PAD=0, UNK=1)
      y: LongTensor scalar label (0/1)
    """
    def __init__(self, infile, word_embeddings, max_len=60):
        self.examples = read_sentiment_examples(infile)
        self.we = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.max_len = max_len

        # Per writeup: PAD index 0, UNK index 1
        self.pad_idx = 0
        self.unk_idx = 1

        self.x_data = []
        self.y_data = []

        for ex in self.examples:
            idxs = []
            for w in ex.words:
                wi = self.word_indexer.index_of(w)
                if wi == -1:
                    wi = self.unk_idx
                idxs.append(wi)

            # pad / truncate to max_len
            if len(idxs) < self.max_len:
                idxs = idxs + [self.pad_idx] * (self.max_len - len(idxs))
            else:
                idxs = idxs[:self.max_len]

            self.x_data.append(torch.tensor(idxs, dtype=torch.long))
            self.y_data.append(torch.tensor(ex.label, dtype=torch.long))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]


class DAN(nn.Module):
    """
    Deep Averaging Network:
      avg(embeddings) -> MLP -> log_softmax (for NLLLoss)
    """
    def __init__(
        self,
        word_embeddings,
        hidden_size=200,
        num_layers=2,
        dropout=0.3,
        frozen_embeddings=False,
        use_glove = True,
        emb_dim = None
    ):
        super().__init__()
        self.pad_idx = 0

        if use_glove:
            # pretrained init (your 1a)
            self.embedding = word_embeddings.get_initialized_embedding_layer(
                frozen=frozen_embeddings
            )
            emb_dim = word_embeddings.get_embedding_length()
        else:
            # 1b: random init
            vocab_size = len(word_embeddings.word_indexer)
            if emb_dim is None:
                emb_dim = word_embeddings.get_embedding_length()  # match glove dim (50 or 300)
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=emb_dim,
                padding_idx=self.pad_idx
            )
            # Embeddings should be trainable for 1b
            self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(dropout)

        layers = []
        in_dim = emb_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.mlp = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        self.out = nn.Linear(in_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        x: LongTensor [batch, max_len]
        """
        x = x.long() 
        emb = self.embedding(x)  # [B, L, D]

        # mask PAD tokens
        mask = (x != self.pad_idx).float()          # [B, L]
        lengths = mask.sum(dim=1).clamp(min=1.0)    # [B]

        summed = (emb * mask.unsqueeze(-1)).sum(dim=1)     # [B, D]
        avg = summed / lengths.unsqueeze(-1)               # [B, D]

        avg = self.dropout(avg)
        h = self.mlp(avg)                                  # [B, hidden] or [B, D]
        logits = self.out(h)                               # [B, 2]
        return self.log_softmax(logits)