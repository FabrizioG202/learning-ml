# This is based on https://www.youtube.com/watch?v=ISNdQcPhsts
# %%

import math

import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        return self.embedding(x) * math.sqrt(self.d_model)


# %%
class PositionalEncodings(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        pe = torch.zeros(seq_len, d_model)

        # initialize the encodings
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos, i] = math.cos(pos / (10000 ** ((i - 1) / d_model)))

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        # x is (batch, seq_len, d_model)
        assert len(x.shape) == 3
        assert x.shape[2] == self.pe.shape[1], (
            "Embedding dimension is not equal to the expected one"
        )
        assert x.shape[1] <= self.pe.shape[0], (
            "True Sequence length is larger than supported max value"
        )

        # The passed in sequence length might
        # be shorter than seq_len
        true_seq_len = x.shape[1]

        # Manual broadcasting (the batch dimension would be
        # automatically handled by pytorch using broadcast
        # but we mak it manual here to be more clear)
        x = x + self.pe[:true_seq_len, :].unsqueeze(0)

        return self.dropout(x)


class AddAndNorm(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):  # ty: ignore[invalid-mutable-override]
        assert len(x.shape) == 3, (
            "Input tensor must be of shape (batch, seq_len, d_model)"
        )

        # For the two items below, we could also use
        # dim=-1, but its equivalent for 3d tensors, which are expected.
        means = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)

        normalized = (x - means) / (torch.sqrt(var + self.eps))

        return self.alpha * normalized + self.bias
