import torch
from torch import nn

from .pooling import *

__all__ = ("PoolingType", "BagOfWords")


class BagOfWords(nn.Module):
    def __init__(self, vocabulary_size: int, encoded_size: int, pooling_type: PoolingType):
        super(BagOfWords, self).__init__()
        self._lookup = nn.Embedding(vocabulary_size, encoded_size)
        self._pool = pooling_functions[pooling_type]

    def forward(self, token_idxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lookups = self._lookup(token_idxs)
        return self._pool(lookups, mask)
