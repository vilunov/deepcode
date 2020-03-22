from typing import Dict

import torch
from torch import nn

from deepcode.encoders import *

__all__ = ("Model",)


class Model(nn.Module):
    def __init__(self, dropout_rate: float, encoders_code: Dict[str, Encoder], encoder_doc: Encoder):
        super().__init__()
        self.encoders_code = encoders_code
        self.encoder_doc = encoder_doc
        for key, value in self.encoders_code.items():
            self.add_module(f"encoders_code[{key}]", value)
        self.add_module("encoder_doc", self.encoder_doc)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, snippet_dict):
        encoded_codes, encoded_docs = [], []
        for language, (code_vec, code_mask, _, _, name_vec, name_mask) in snippet_dict.items():
            encoder_code = self.encoders_code[language]
            encoded_codes.append(encoder_code(code_vec, code_mask))
            encoded_docs.append(self.encoder_doc(name_vec, name_mask))
        return self.dropout(torch.cat(encoded_codes)), self.dropout(torch.cat(encoded_docs))
