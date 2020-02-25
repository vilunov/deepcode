import torch
from torch import nn

from deepcode import config
from deepcode.config import PoolingType
from deepcode.encoders.pooling import *

__all__ = ("BagOfWords", "Encoder")


class Encoder(nn.Module):
    @staticmethod
    def from_config(config: config.Encoder, encoded_size: int) -> "Encoder":
        if config.type == "nbow":
            if config.pooling_type is None:
                raise ValueError("Pooling type is required for neural bag of word encoders")
            return BagOfWords(32767, encoded_size, config.pooling_type)
        else:
            raise ValueError(f"Unknown encoder type: {config.type}")


class BagOfWords(Encoder):
    def __init__(self, vocabulary_size: int, encoded_size: int, pooling_type: PoolingType):
        super(BagOfWords, self).__init__()
        self._lookup = nn.Embedding(vocabulary_size, encoded_size)
        self._pool = pooling_functions[pooling_type]

    def forward(self, token_idxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lookups = self._lookup(token_idxs)
        return self._pool(lookups, mask)
