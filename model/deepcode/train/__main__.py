from typing import Tuple, Iterable

import torch
import h5py
from torch import Tensor

from deepcode.encoders import *

__all__ = ("main",)

Datapoint = Tuple[Tensor, Tensor, Tensor, Tensor]


def convert(dataset_entry) -> Datapoint:
    code_len, code_vec, doc_len, doc_vec = dataset_entry
    code_vec = torch.tensor(code_vec.astype("int64"))
    doc_vec = torch.tensor(doc_vec.astype("int64"))
    code_mask = torch.zeros(code_vec.shape, dtype=bool)
    doc_mask = torch.zeros(doc_vec.shape, dtype=bool)
    code_mask[:code_len] = True
    doc_mask[:doc_len] = True
    return code_vec, code_mask, doc_vec, doc_mask


def par_stack(entries: Iterable[Datapoint]):
    entries = list(zip(*entries))
    result = tuple(torch.stack(t) for t in entries)
    return result


def main():
    dataset = h5py.File("cache/data/code.h5")
    data_go = dataset["go"]
    idxs = list(range(10))
    entries = [convert(data_go[i]) for i in idxs]
    code_vec, code_mask, doc_vec, doc_mask = par_stack(entries)
    encoder_go = BagOfWords(
        vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MAX
    )
    encoded = encoder_go(code_vec, code_mask)
    pass


if __name__ == "__main__":
    main()
