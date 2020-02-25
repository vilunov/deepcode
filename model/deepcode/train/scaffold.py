import torch
from torch import optim
import numpy as np
import logging

from deepcode.loss import *
from deepcode.model import Model
from deepcode.prefetcher import BackgroundGenerator

__all__ = ("Scaffold",)


class Scaffold:
    def __init__(self, arguments):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._init_data(arguments.batch_size)
        self.model = Model(
            languages=self.train_counts.keys(), dropout_rate=arguments.dropout, repr_size=arguments.repr_size,
        ).to(self.device)
        if arguments.weights_path is not None:
            self.model.load_state_dict(torch.load(arguments.weights_path, map_location=self.device))
            logging.info(f"Loaded weights from file {arguments.weights_path}")
        self.__optimizer = optim.Adam(self.model.parameters(), lr=arguments.learning_rate)
        if arguments.loss == "triplet":
            if arguments.loss_margin is None:
                raise ValueError("Loss margin required for triplet loss")
            self.__loss = LossTriplet(arguments.loss_margin)
        elif arguments.loss == "crossentropy":
            self.__loss = LossCrossEntropy()
        else:
            raise ValueError(f"Incorrect loss type: {arguments.loss}, expected triplet or crossentropy")

    def _init_data(self, batch_size):
        from .data import open_data

        self.train_counts, self.train_data, self._train_file = open_data(
            "../cache/data/train.h5", self.device, batch_size
        )
        self.valid_counts, self.valid_data, self._valid_file = open_data(
            "../cache/data/valid.h5", self.device, batch_size
        )

    def epoch_train(self, tqdm):
        self.model.train(True)
        pbar = tqdm(desc="Training", total=len(self.train_data))
        losses = np.zeros(len(self.train_data))
        train_data = BackgroundGenerator(self.train_data)
        for iteration, snippet_dict in enumerate(train_data):
            encoded_code, encoded_doc = self.model(snippet_dict)
            self.__optimizer.zero_grad()
            loss = self.__loss(encoded_code, encoded_doc)
            loss.backward()
            self.__optimizer.step()
            pbar.update()

            losses[iteration] = loss.item()
            stats = {
                "loss_min": losses[: iteration + 1].min(),
                "loss_max": losses[: iteration + 1].max(),
                "loss_std": losses[: iteration + 1].std(),
                "loss_mean": losses[: iteration + 1].mean(),
                "loss_last": losses[iteration],
            }
            pbar.set_postfix(**stats)
        pbar.close()

    def epoch_validate(self, tqdm):
        self.model.train(False)
        mrr_sum, mrr_num = 0.0, 0
        pbar = tqdm(desc="Validation", total=len(self.valid_data))
        for snippet_dict in self.valid_data:
            encoded_code, encoded_doc = self.model(snippet_dict)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            weights = distances.diag()
            mrr_sum += (distances >= weights.unsqueeze(1)).sum(dim=1).type(torch.float32).reciprocal().mean().item()
            mrr_num += 1
            pbar.update()
            pbar.set_postfix(mrr=mrr_sum / mrr_num)
        pbar.close()
        return {"mrr": mrr_sum / mrr_num}

    def close(self):
        self._train_file.close()
        self._valid_file.close()
