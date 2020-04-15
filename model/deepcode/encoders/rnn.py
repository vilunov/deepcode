import torch as t
from torch import nn

from deepcode.config import PoolingType
from deepcode.encoders import Encoder
from deepcode.encoders.pooling import pooling_functions

__all__ = ("LSTM",)


class LSTM(Encoder):
    def __init__(self, *, intermediate_size: int, encoded_size: int, vocab_size: int, pooling_type: PoolingType):
        super().__init__()
        self._lookup = nn.Embedding(num_embeddings=vocab_size, embedding_dim=intermediate_size)
        self._rnn = nn.LSTM(
            input_size=intermediate_size, hidden_size=encoded_size, num_layers=1, batch_first=True, bidirectional=False
        )
        self._pool = pooling_functions[pooling_type]

    def forward(self, token_idxs: t.Tensor, mask: t.Tensor) -> t.Tensor:
        x = self._lookup(token_idxs)
        x, _ = self._rnn(x)
        return self._pool(x, mask)
