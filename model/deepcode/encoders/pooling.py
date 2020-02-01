from enum import Enum, auto
from typing import Dict, Callable

import torch
from torch import Tensor

__all__ = ("PoolingType", "pooling_functions")


def mean(tensor: Tensor, mask: Tensor) -> Tensor:
    tensor = tensor.clone()
    tensor[~mask] = 0
    tensor = tensor.sum(-2)
    mask = mask.sum(-1)
    return tensor / mask.unsqueeze(1)


def max(tensor: Tensor, mask: Tensor) -> Tensor:
    tensor = tensor.clone()
    tensor[~mask] = float("-inf")
    return torch.max(tensor, -2).values


class PoolingType(Enum):
    MEAN = auto()
    MAX = auto()


pooling_functions: Dict[PoolingType, Callable[[Tensor, Tensor], Tensor]] = {
    PoolingType.MEAN: mean,
    PoolingType.MAX: max,
}
