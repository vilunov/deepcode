import logging
from abc import ABC
from typing import Optional

import torch

from deepcode.config import Config
from deepcode.model import Model, Encoder

__all__ = ("AbstractScaffold",)


class AbstractScaffold(ABC):
    def __init__(self, config: Config, weights_path: Optional[str]):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Model(
            dropout_rate=config.training.dropout_rate,
            encoders_code={
                language: Encoder.from_config(encoder_config, config.model.encoded_dims)
                for language, encoder_config in config.model.code_encoder.items()
            },
            encoder_doc=Encoder.from_config(config.model.doc_encoder, config.model.encoded_dims),
        ).to(self.device)
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logging.info(f"Loaded weights from file {weights_path}")
