import torch

from deepcode.encoders.pooling import *


def test_mean():
    func = pooling_functions[PoolingType.MEAN]
    tensor = torch.tensor([[[0.0, 0.5, 0.3], [-1.0, -0.7, 0.1]]])
    assert tensor.shape == (1, 2, 3)
    mask = torch.tensor([[True, True]])
    assert mask.shape == (1, 2)
    result = func(tensor, mask)
    assert result.shape == (1, 3,)
    expected = torch.tensor([[-0.5, -0.1, 0.2]])
    assert (result - expected).sum().abs() < 1e-4


def test_max():
    func = pooling_functions[PoolingType.MAX]
    tensor = torch.tensor([[[0.0, 0.5, 0.3, -0.5], [-1.0, -0.7, 0.1, -0.3]]])
    assert tensor.shape == (1, 2, 4)
    mask = torch.tensor([[True, True]])
    assert mask.shape == (1, 2)
    result = func(tensor, mask)
    assert result.shape == (1, 4,)
    expected = torch.tensor([0.0, 0.5, 0.3, -0.3])
    assert (result - expected).abs().mean() < 1e-7


def test_mean_mask():
    func = pooling_functions[PoolingType.MEAN]
    tensor = torch.tensor([[[0.0, 0.5, 0.3], [-1.0, -0.7, 0.1]]])
    assert tensor.shape == (1, 2, 3)
    mask = torch.tensor([[True, False]])
    assert mask.shape == (1, 2)
    result = func(tensor, mask)
    assert result.shape == (1, 3,)
    expected = torch.tensor([0.0, 0.5, 0.3])
    assert (result - expected).sum().abs() < 1e-4
