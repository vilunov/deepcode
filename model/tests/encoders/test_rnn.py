import torch

from deepcode.encoders import PoolingType
from deepcode.encoders.rnn import LSTM


def test_lstm_mean():
    encoder = LSTM(vocab_size=128, encoded_size=64, intermediate_size=96, pooling_type=PoolingType.MEAN)
    input_batch = torch.tensor([[1, 2, 3, 4], [4, 3, 1, 2]])
    assert input_batch.shape == (2, 4)
    input_mask = torch.ones((2, 4)).type(torch.bool)

    output_batch = encoder(input_batch, input_mask)
    assert output_batch.shape == (2, 64)
    assert isinstance(output_batch, torch.Tensor)


if __name__ == "__main__":
    test_lstm_mean()
