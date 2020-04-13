from typing import Dict, Callable

import torch as t
from torch import Tensor

from deepcode.config import PoolingType

__all__ = ("pooling_functions",)


def mean(tensor: Tensor, mask: Tensor) -> Tensor:
    tensor = tensor.clone()
    tensor[~mask] = 0
    tensor = tensor.sum(-2)
    mask = mask.sum(-1)
    return tensor / mask.unsqueeze(1).type(tensor.dtype)


def max(tensor: Tensor, mask: Tensor) -> Tensor:
    tensor = tensor.clone()
    tensor[~mask] = float("-inf")
    return tensor.max(dim=-2)[0]


pooling_functions: Dict[PoolingType, Callable[[Tensor, Tensor], Tensor]] = {
    PoolingType.MEAN: mean,
    PoolingType.MAX: max,
}
