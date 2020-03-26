from typing import Dict, Tuple

import torch
from torch import nn, Tensor

from deepcode.encoders import *

__all__ = ("Model", "SnippetInput")

SnippetInput = Tuple[Tensor, Tensor, Tensor, Tensor]


class Model(nn.Module):
    def __init__(self, dropout_rate: float, encoders_code: Dict[str, Encoder], encoder_doc: Encoder):
        super().__init__()
        self.encoders_code = encoders_code
        self.encoder_doc = encoder_doc
        for key, value in self.encoders_code.items():
            self.add_module(f"encoders_code[{key}]", value)
        self.add_module("encoder_doc", self.encoder_doc)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, snippet_dict: Dict[str, SnippetInput]):
        encoded_codes, encoded_docs = [], []
        for language, (code_vec, code_mask, doc_vec, doc_mask) in snippet_dict.items():
            encoder_code = self.encoders_code[language]
            encoded_codes.append(encoder_code(code_vec, code_mask))
            encoded_docs.append(self.encoder_doc(doc_vec, doc_mask))
        return self.dropout(torch.cat(encoded_codes)), self.dropout(torch.cat(encoded_docs))
