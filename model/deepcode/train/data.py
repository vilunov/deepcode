from typing import Tuple, Iterable, Dict, List

import h5py
import torch
from torch import Tensor
from torch.utils.data import (
    Dataset,
    RandomSampler,
    BatchSampler,
    DataLoader,
    ConcatDataset,
)

__all__ = ("Datapoint", "CodeDataset", "open_data")

Datapoint = Tuple[Tensor, Tensor, Tensor, Tensor]


class CodeDataset(Dataset):
    def __init__(self, hdf5_dataset, language, device):
        self.__data = hdf5_dataset
        self.__device = device
        self.__language = language

    def __convert(self, dataset_entry) -> Tuple[str, Datapoint]:
        code_len, code_vec, doc_len, doc_vec = dataset_entry
        code_vec = torch.tensor(code_vec.astype("int64"), device=self.__device)
        doc_vec = torch.tensor(doc_vec.astype("int64"), device=self.__device)
        code_mask = torch.zeros(code_vec.shape, dtype=torch.bool, device=self.__device)
        doc_mask = torch.zeros(doc_vec.shape, dtype=torch.bool, device=self.__device)
        code_mask[:code_len] = True
        doc_mask[:doc_len] = True
        return self.__language, (code_vec, code_mask, doc_vec, doc_mask)

    def __getitem__(self, item):
        return self.__convert(self.__data[item])

    def __len__(self):
        return self.__data.__len__()


def par_stack(entries: Iterable[Datapoint]):
    entries = list(zip(*entries))
    result = tuple(torch.stack(t) for t in entries)
    return result


def collate(entries: Iterable[Tuple[str, Datapoint]]) -> Dict[str, Tensor]:
    from collections import defaultdict

    result: Dict[str, List[Datapoint]] = defaultdict(list)
    for language, datapoint in entries:
        result[language].append(datapoint)
    return {language: par_stack(entries) for language, entries in result.items()}


def open_data(path: str, device: str, batch_size: int) -> Tuple[Dict[str, int], DataLoader, h5py.File]:
    h5_file = h5py.File(path, "r")
    counts: Dict[str, int] = {}
    datasets: List[CodeDataset] = []
    for key in h5_file.keys():
        dataset = CodeDataset(h5_file[key], key, device)
        datasets.append(dataset)
        counts[key] = dataset.__len__()
    data = ConcatDataset(datasets)
    sampler = BatchSampler(RandomSampler(data), batch_size=batch_size, drop_last=True)
    return counts, DataLoader(data, batch_sampler=sampler, collate_fn=collate), h5_file
