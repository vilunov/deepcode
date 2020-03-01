from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

import toml
from dacite import from_dict, Config as DaciteConfig

__all__ = ("PoolingType", "Training", "Model", "Encoder", "Config", "parse_config")


class PoolingType(Enum):
    MEAN = "mean"
    MAX = "max"


@dataclass
class Training:
    learning_rate: float
    data_train: str
    data_valid: str
    loss_type: str
    loss_margin: Optional[float]
    dropout_rate: float
    batch_size_train: int
    batch_size_valid: int


@dataclass
class Encoder:
    type: str
    pooling_type: Optional[PoolingType]


@dataclass
class Model:
    encoded_dims: int
    doc_encoder: Encoder
    code_encoder: Dict[str, Encoder]


@dataclass
class Config:
    training: Training
    model: Model


def parse_config(text: str) -> Config:
    loaded = toml.loads(text)
    return from_dict(data_class=Config, data=loaded, config=DaciteConfig(cast=[PoolingType]))
