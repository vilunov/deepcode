from abc import ABCMeta

import torch
from torch.nn.functional import pairwise_distance, relu
from torch.nn.modules import CrossEntropyLoss

__all__ = ("Loss", "LossCrossEntropy", "LossTriplet")


class Loss(metaclass=ABCMeta):
    def __call__(self, encoded_code: torch.Tensor, encoded_doc: torch.Tensor):
        raise NotImplementedError


class LossCrossEntropy(Loss):
    def __init__(self):
        self.__criterion = CrossEntropyLoss()

    def __call__(self, encoded_code: torch.Tensor, encoded_doc: torch.Tensor):
        distances = torch.matmul(encoded_code, encoded_doc.T)
        labels = torch.arange(0, encoded_code.shape[0], device=encoded_code.device)
        return self.__criterion(distances, labels)


class LossTriplet(Loss):
    def __init__(self, margin: float = 1.0):
        self.__margin = margin

    def __call__(self, encoded_code: torch.Tensor, encoded_doc: torch.Tensor):
        positive = pairwise_distance(encoded_code, encoded_doc, 2, 1e-6)
        rolled_encoded_code = torch.cat((encoded_code[1:], encoded_code[:1]))
        negative = pairwise_distance(rolled_encoded_code, encoded_doc, 2, 1e-6)
        return relu(positive - negative + self.__margin).max()
