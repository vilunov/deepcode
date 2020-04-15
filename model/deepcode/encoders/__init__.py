import torch as t
from torch import nn
import numpy as np

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
            if config.weights_path is not None:
                weights = np.fromfile(config.weights_path).reshape(-1, encoded_size)
                weights = t.tensor(weights)
                return BagOfWords.from_weights(weights, config.pooling_type)
            else:
                if config.vocabulary_size is None:
                    raise ValueError("Vocabulary size is required for neural bag of word encoders")
                return BagOfWords.new(config.vocabulary_size, encoded_size, config.pooling_type)
        elif config.type == "lstm":
            from deepcode.encoders.rnn import LSTM

            if config.pooling_type is None:
                raise ValueError("Pooling type is required for LSTM encoders")
            if config.vocabulary_size is None:
                raise ValueError("Vocabulary size is required for LSTM encoders")
            if config.intermediate_size is None:
                raise ValueError("Intermediate vector size is required for LSTM encoders")
            return LSTM(
                intermediate_size=config.intermediate_size,
                encoded_size=encoded_size,
                pooling_type=config.pooling_type,
                vocab_size=config.vocabulary_size,
            )
        else:
            raise ValueError(f"Unknown encoder type: {config.type}")


class BagOfWords(Encoder):
    def __init__(self, lookup: nn.Embedding, pooling_type: PoolingType):
        super(BagOfWords, self).__init__()
        self._lookup = lookup
        self._pool = pooling_functions[pooling_type]

    @staticmethod
    def new(vocabulary_size: int, encoded_size: int, pooling_type: PoolingType):
        lookup = nn.Embedding(vocabulary_size, encoded_size)
        return BagOfWords(lookup, pooling_type)

    @staticmethod
    def from_weights(weights: t.Tensor, pooling_type: PoolingType):
        lookup = nn.Embedding.from_pretrained(weights.type(t.float32), freeze=True)
        return BagOfWords(lookup, pooling_type)

    def forward(self, token_idxs: t.Tensor, mask: t.Tensor) -> t.Tensor:
        lookups = self._lookup(token_idxs)
        return self._pool(lookups, mask)
