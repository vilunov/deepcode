import torch
from torch import optim

from deepcode.loss import *
from deepcode.model import Model

__all__ = ("Scaffold",)


class Scaffold:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._init_data()
        self.model = Model(self.train_counts.keys()).to(self.device)
        self.__optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.__loss = LossTriplet()

    def _init_data(self):
        from .data import open_data

        self.train_counts, self.train_data, self._train_file = open_data("../cache/data/train.h5", self.device)
        self.valid_counts, self.valid_data, self._valid_file = open_data("../cache/data/valid.h5", self.device)

    def epoch_train(self, tqdm):
        self.model.train(True)
        pbar = tqdm(desc="Training", total=len(self.train_data))
        for snippet_dict in self.train_data:
            encoded_code, encoded_doc = self.model(snippet_dict)
            self.__optimizer.zero_grad()
            loss = self.__loss(encoded_code, encoded_doc)
            loss.backward()
            self.__optimizer.step()
            pbar.update()
            pbar.set_postfix(loss=loss.item())
        pbar.close()

    def epoch_validate(self, tqdm):
        self.model.train(False)
        mrr_sum, mrr_num = 0.0, 0
        pbar = tqdm(desc="Validation", total=len(self.valid_data))
        for snippet_dict in self.valid_data:
            encoded_code, encoded_doc = self.model(snippet_dict)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            weights = distances.diag()
            mrr_sum += (distances >= weights).sum(dim=1).type(torch.float32).reciprocal().mean().item()
            mrr_num += 1
            pbar.update()
            pbar.set_postfix(mrr=mrr_sum / mrr_num)
        pbar.close()
        return {"mrr": mrr_sum / mrr_num}

    def close(self):
        self._train_file.close()
        self._valid_file.close()
