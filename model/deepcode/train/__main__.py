from typing import Tuple, Iterable

import torch
import h5py
from torch import optim, Tensor
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import Dataset, RandomSampler, BatchSampler, DataLoader
from tqdm import tqdm

from deepcode.encoders import *

__all__ = ("main",)

Datapoint = Tuple[Tensor, Tensor, Tensor, Tensor]

class CodeDataset(Dataset):
    def __init__(self, hdf5_dataset, device):
        self.__data = hdf5_dataset
        self.__device = device

    def __convert(self, dataset_entry) -> Datapoint:
        code_len, code_vec, doc_len, doc_vec = dataset_entry
        code_vec = torch.tensor(code_vec.astype("int64"), device=self.__device)
        doc_vec = torch.tensor(doc_vec.astype("int64"), device=self.__device)
        code_mask = torch.zeros(code_vec.shape, dtype=torch.bool, device=self.__device)
        doc_mask = torch.zeros(doc_vec.shape, dtype=torch.bool, device=self.__device)
        code_mask[:code_len] = True
        doc_mask[:doc_len] = True
        return code_vec, code_mask, doc_vec, doc_mask


    def __getitem__(self, item):
        return self.__convert(self.__data[item])

    def __len__(self):
        return self.__data.__len__()



def par_stack(entries: Iterable[Datapoint]):
    entries = list(zip(*entries))
    result = tuple(torch.stack(t) for t in entries)
    return result


def parameters(*encoders):
    for e in encoders:
        for p in e.parameters():
            yield p


def open_data(path: str, device: str):
    dataset = h5py.File(path, "r")
    data_go = CodeDataset(dataset["go"], device)
    sampler = BatchSampler(RandomSampler(data_go), batch_size=512, drop_last=True)
    return DataLoader(data_go, batch_sampler=sampler, collate_fn=par_stack)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data = open_data("../cache/data/train.h5", device)
    valid_data = open_data("../cache/data/valid.h5", device)

    encoder_go = BagOfWords(
        vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN
    ).to(device)
    encoder_doc = BagOfWords(
        vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN
    ).to(device)
    optimizer = optim.Adam(parameters(encoder_go, encoder_doc), lr=1e-2)
    criterion = CrossEntropyLoss()
    for epoch in range(100):
        encoder_go.train(True)
        encoder_doc.train(True)
        for code_vec, code_mask, doc_vec, doc_mask in tqdm(train_data, desc="Training"):
            labels = torch.arange(0, code_vec.shape[0], device=device)
            optimizer.zero_grad()
            encoded_code = encoder_go(code_vec, code_mask)
            encoded_doc = encoder_doc(doc_vec, doc_mask)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            loss = criterion(distances, labels)
            loss.backward()
            optimizer.step()
            # print("Loss:", loss.item())
        encoder_go.train(False)
        encoder_doc.train(False)
        mrr_sum = 0
        mrr_num = 0
        for code_vec, code_mask, doc_vec, doc_mask in tqdm(valid_data, desc="Validation"):
            encoded_code = encoder_go(code_vec, code_mask)
            encoded_doc = encoder_doc(doc_vec, doc_mask)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            weights = distances.diag()
            mrr_sum += (1.0 / (distances >= weights).sum(dim=1).type(torch.float32)).mean().item()
            mrr_num += 1
        print(mrr_sum / mrr_num)
    pass


if __name__ == "__main__":
    main()
