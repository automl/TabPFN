import math

import torch
from torch import nn


# Protocol for positonal encodings.
# __init__(d_model, max_len=..[, more optionals])
# forward(x: (seq_len, bs, d_model)) -> Tensor of shape (*x.shape[:2],d_model) containing pos. embeddings


class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super(NoPositionalEncoding, self).__init__()
        pass

    def forward(self, x):
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :] + x # * math.sqrt(x.shape[-1])
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.max_seq_len = max_len
        #self.positional_embeddings = nn.Embedding(max_len, d_model)
        self.positional_embeddings = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.positional_embeddings, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), 'seq_len can be at most max_len.'
        pos_emb = self.positional_embeddings[:seq_len]
        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x #* math.sqrt(x.shape[-1])


class PairedScrambledPositionalEncodings(LearnedPositionalEncoding):
    # TODO check whether it is a problem to use the same perm. for full batch
    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), 'seq_len can be at most max_len.'
        assert len(self.positional_embeddings) % 2 == 0, 'Please specify an even max_len.'

        paired_embs = self.positional_embeddings.view(len(self.positional_embeddings), -1, 2)
        pos_emb = paired_embs[torch.randperm(len(paired_embs))].view(*self.positional_embeddings.shape)[:seq_len]

        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x #* math.sqrt(x.shape[-1])








