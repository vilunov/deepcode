import torch
from torch import nn

from deepcode.encoders import *

__all__ = ("Model",)


class Model(nn.Module):
    def __init__(self, languages):
        super().__init__()
        self.encoders_code = {
            language: BagOfWords(vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN)
            for language in languages
        }
        self.encoder_doc = BagOfWords(vocabulary_size=32767, encoded_size=128, pooling_type=PoolingType.MEAN)

        for key, value in self.encoders_code.items():
            self.add_module(f"encoders_code[{key}]", value)
        self.add_module("encoder_doc", self.encoder_doc)

    def forward(self, snippet_dict):
        encoded_codes, encoded_docs = [], []
        for language, (code_vec, code_mask, doc_vec, doc_mask) in snippet_dict.items():
            encoder_code = self.encoders_code[language]
            encoded_codes.append(encoder_code(code_vec, code_mask))
            encoded_docs.append(self.encoder_doc(doc_vec, doc_mask))
        return torch.cat(encoded_codes), torch.cat(encoded_docs)
