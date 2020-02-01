import torch

from deepcode import encoders


def test_bow_mean():
    encoder = encoders.BagOfWords(128, 64, encoders.PoolingType.MEAN)
    input_batch = torch.tensor([[1, 2, 3, 4], [4, 3, 1, 2]])
    assert input_batch.shape == (2, 4)
    input_mask = torch.ones((2, 4)).type(torch.bool)

    output_batch = encoder(input_batch, input_mask)
    assert output_batch.shape == (2, 64)
    assert isinstance(output_batch, torch.Tensor)
    assert (output_batch[0] - output_batch[1]).abs().mean() < 1e-7


def test_bow_max():
    encoder = encoders.BagOfWords(128, 64, encoders.PoolingType.MAX)
    input_batch = torch.tensor([[1, 2, 3, 4], [4, 3, 1, 2]])
    assert input_batch.shape == (2, 4)
    input_mask = torch.ones((2, 4)).type(torch.bool)
    output_batch = encoder(input_batch, input_mask)

    assert output_batch.shape == (2, 64)
    assert isinstance(output_batch, torch.Tensor)
    assert (output_batch[0] - output_batch[1]).abs().mean() < 1e-7


def test_bow_mean_mask():
    encoder = encoders.BagOfWords(128, 64, encoders.PoolingType.MEAN)
    input_batch = torch.tensor([[1, 2, 3, 4], [4, 3, 1, 2], [1, 2, 4, 3]])
    assert input_batch.shape == (3, 4)
    input_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]).type(
        torch.bool
    )
    assert input_batch.shape == (3, 4)
    output_batch = encoder(input_batch, input_mask)

    assert output_batch.shape == (3, 64)
    assert isinstance(output_batch, torch.Tensor)
    assert (output_batch[2] - output_batch[0] - output_batch[1]).abs().mean() > 1e-7
