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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):  # ty: ignore[invalid-mutable-override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        dropout: nn.Dropout | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert len(q.shape) == 4, (
            "Query tensor must be of shape (batch, num_heads, seq_len, d_k)"
        )

        assert q.shape == k.shape == v.shape, (
            "Query, Key and Value tensors must have the same shape"
        )

        d_k = q.shape[-1]

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask before softmax
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_probs)

        return attention_scores @ v, attention_scores

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ):
        q_prime: torch.Tensor = self.w_q(q)
        k_prime: torch.Tensor = self.w_k(k)
        v_prime: torch.Tensor = self.w_v(v)

        # split q into multiple heads
        q_prime = q_prime.view(
            q_prime.shape[0], q_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        k_prime = k_prime.view(
            k_prime.shape[0], k_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        v_prime = v_prime.view(
            v_prime.shape[0], v_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = self.attention(
            q_prime, k_prime, v_prime, mask, self.dropout
        )

        # concat x back into a single 3d matrix
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)
        return self.w_o(x)