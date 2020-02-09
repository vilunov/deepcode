import torch
from torch import optim
from torch.nn.modules import CrossEntropyLoss

from deepcode.encoders import *

__all__ = ("Scaffold",)


class Scaffold:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._open_data()
        self._init_encoders()
        self._init_optimizer()

    def _open_data(self):
        from .data import open_data

        self.train_counts, self.train_data, self._train_file = open_data("../cache/data/train.h5", self.device)
        self.valid_counts, self.valid_data, self._valid_file = open_data("../cache/data/valid.h5", self.device)

    def _init_encoders(self):
        device = self.device
        self.encoders_code = {
            language: BagOfWords(vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN).to(device)
            for language in self.train_counts.keys()
        }
        self.encoder_doc = BagOfWords(vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN).to(device)

    def _parameters(self):
        for e in self.encoders_code.values():
            for p in e.parameters():
                yield p
        for p in self.encoder_doc.parameters():
            yield p

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self._parameters(), lr=1e-2)
        self.criterion = CrossEntropyLoss()

    def train(self, mode: bool):
        for e in self.encoders_code.values():
            e.train(mode)
        self.encoder_doc.train(mode)

    def forward(self, snippet_dict):
        encoded_codes, encoded_docs = [], []
        for language, (code_vec, code_mask, doc_vec, doc_mask) in snippet_dict.items():
            encoder_code = self.encoders_code[language]
            encoded_codes.append(encoder_code(code_vec, code_mask))
            encoded_docs.append(self.encoder_doc(doc_vec, doc_mask))
        return torch.cat(encoded_codes), torch.cat(encoded_docs)

    def epoch_train(self, pbar):
        self.train(True)
        pbar.total = len(self.train_data)
        for snippet_dict in self.train_data:
            encoded_code, encoded_doc = self.forward(snippet_dict)
            labels = torch.arange(0, encoded_code.shape[0], device=self.device)
            self.optimizer.zero_grad()
            distances = torch.matmul(encoded_code, encoded_doc.T)
            loss = self.criterion(distances, labels)
            loss.backward()
            self.optimizer.step()
            pbar.update()

    def epoch_validate(self, pbar):
        self.train(False)
        pbar.total = len(self.valid_data)
        mrr_sum = 0
        mrr_num = 0
        for snippet_dict in self.valid_data:
            encoded_code, encoded_doc = self.forward(snippet_dict)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            weights = distances.diag()
            mrr_sum += (distances >= weights).sum(dim=1).type(torch.float32).reciprocal().mean().item()
            mrr_num += 1
            pbar.update()
        return mrr_sum / mrr_num

    def close(self):
        self._train_file.close()
        self._valid_file.close()
