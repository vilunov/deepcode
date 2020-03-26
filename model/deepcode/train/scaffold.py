from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.distributions.binomial import Binomial

from deepcode.config import Config
from deepcode.train.loss import *
from deepcode.prefetcher import BackgroundGenerator
from deepcode.scaffold import AbstractScaffold

__all__ = ("TrainScaffold",)


class TrainScaffold(AbstractScaffold):
    def __init__(self, config: Config, weights_path: Optional[str]):
        super().__init__(config, weights_path)
        self._init_data(config)
        self.__optimizer = optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        if config.training.loss_type == "triplet":
            if config.training.loss_margin is None:
                raise ValueError("Loss margin required for triplet loss")
            self.__loss = LossTriplet(config.training.loss_margin)
        elif config.training.loss_type == "crossentropy":
            self.__loss = LossCrossEntropy()
        else:
            raise ValueError(f"Incorrect loss type: {config.training.loss_type}, expected triplet or crossentropy")
        self.__choice_distribution = Binomial(probs=0.1)

    def stochastic_choice(self, doc_vec, doc_mask, name_vec, name_mask):
        choices = self.__choice_distribution.sample((doc_vec.size(0),)).type(torch.bool)
        name_vec_pad = torch.zeros(
            name_vec.size(0), doc_vec.size(1) - name_vec.size(1), dtype=name_vec.dtype, device=self.device
        )
        name_mask_pad = torch.zeros(
            name_mask.size(0), doc_mask.size(1) - name_mask.size(1), dtype=name_mask.dtype, device=self.device
        )
        name_vec = torch.cat([name_vec, name_vec_pad], dim=1)
        name_mask = torch.cat([name_mask, name_mask_pad], dim=1)
        doc_vec[choices] = name_vec[choices]
        doc_mask[choices] = name_mask[choices]
        return doc_vec, doc_mask

    def _init_data(self, config: Config):
        import h5py
        from torch.utils.data import DataLoader, RandomSampler, BatchSampler, ConcatDataset
        from deepcode.train import CodeDataset
        from deepcode.train.data import open_data, collate

        batch_t = config.training.batch_size_train
        batch_v = config.training.batch_size_valid
        path_train = config.training.data_train
        path_valid = config.training.data_valid
        languages = set(config.model.code_encoder.keys())
        _, self.train_data, self._train_file = open_data(path_train, self.device, batch_t, languages)

        valid_datasets = dict()
        self.valid_data = dict()
        self._valid_file = h5py.File(path_valid, "r")
        for lang in languages:
            data = CodeDataset(self._valid_file[lang], lang, self.device)
            sampler = BatchSampler(RandomSampler(data), batch_size=batch_v, drop_last=True)
            valid_datasets[lang] = data
            self.valid_data[lang] = DataLoader(data, batch_sampler=sampler, collate_fn=collate)
        whole_valid_dataset = ConcatDataset(list(valid_datasets.values()))
        all_sampler = BatchSampler(RandomSampler(whole_valid_dataset), batch_size=batch_v, drop_last=True)
        dataloader = DataLoader(whole_valid_dataset, batch_sampler=all_sampler, collate_fn=collate)
        self.valid_data["all"] = dataloader

    def epoch_train(self, tqdm):
        self.model.train(True)
        pbar = tqdm(desc="Training", total=len(self.train_data))
        losses = np.zeros(len(self.train_data))
        train_data = BackgroundGenerator(self.train_data)
        for iteration, snippet_dict in enumerate(train_data):
            model_input = {
                key: (code_vec, code_mask) + self.stochastic_choice(doc_vec, doc_mask, name_vec, name_mask)
                for key, (code_vec, code_mask, doc_vec, doc_mask, name_vec, name_mask) in snippet_dict.items()
            }
            encoded_code, encoded_doc = self.model(model_input)
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
        mrr = dict()
        for name in self.valid_data.keys():
            mrr[name] = self.calculate_mrr(tqdm, name)
        return {"mrr": mrr}

    def calculate_mrr(self, tqdm, name: str):
        mrr_sum, mrr_num = 0.0, 0
        pbar = tqdm(desc=f"Calculating MRR of {name}", total=len(self.valid_data[name]))
        for snippet_dict in self.valid_data[name]:
            model_input = {
                key: (code_vec, code_mask) + self.stochastic_choice(doc_vec, doc_mask, name_vec, name_mask)
                for key, (code_vec, code_mask, doc_vec, doc_mask, name_vec, name_mask) in snippet_dict.items()
            }
            encoded_code, encoded_doc = self.model(model_input)
            distances = torch.matmul(encoded_code, encoded_doc.T)
            weights = distances.diag()
            mrr_sum += (distances >= weights.unsqueeze(1)).sum(dim=1).type(torch.float32).reciprocal().mean().item()
            mrr_num += 1
            pbar.update()
            pbar.set_postfix(mrr=mrr_sum / mrr_num)
        pbar.close()
        return mrr_sum / mrr_num

    def close(self):
        self._train_file.close()
        self._valid_file.close()
