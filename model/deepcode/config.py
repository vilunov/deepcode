from dataclasses import dataclass
from typing import Dict

import toml
from dacite import from_dict

__all__ = ("Language", "Config", "parse_config")


@dataclass
class Language:
    train_path: str
    valid_path: str


@dataclass
class Config:
    languages: Dict[str, Language]


def parse_config(text: str) -> Config:
    loaded = toml.loads(text)
    return from_dict(Config, loaded)
