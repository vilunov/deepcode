import argparse
import csv
import gzip
import json
import logging

import h5py
import pandas as pd
import torch as t
import wandb
from annoy import AnnoyIndex

from deepcode.config import parse_config
from deepcode.predict.scaffold import PredictScaffold
from deepcode.predict.data import open_data, CodeDataset


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--queries", type=str, required=True, dest="queries", help="input queries file")
    parser.add_argument("-d", "--data", type=str, required=True, dest="data", help="input metadata files", nargs="+")
    parser.add_argument("-t", "--tokens", type=str, required=True, dest="tokens", help="input tokens file", nargs="+")
    parser.add_argument("-o", "--output", type=str, required=True, dest="output", help="output predictions file")
    parser.add_argument("-m", "--model", type=str, required=True, dest="model", help="model weights file")
    parser.add_argument("-c", "--config", type=str, required=True, dest="config", help="model configuration file")
    args = parser.parse_args()
    print(args)
    return args


class LangIndex:
    def __init__(self, repr_size: int, index_type: str):
        super().__init__()
        self.index = AnnoyIndex(repr_size, index_type)
        self.func_names = []
        self.urls = []

    def __len__(self):
        return len(self.func_names)

    def __getitem__(self, item):
        return self.func_names[item], self.urls[item]

    def append(self, repr, func_name, url):
        idx = len(self)
        self.index.add_item(idx, repr)
        self.func_names.append(func_name)
        self.urls.append(url)


def main(enable_wandb: bool = True):
    logging.basicConfig(level=logging.INFO)
    args = arguments()
    if enable_wandb:
        wandb.init(name="test")
    with open(args.config, "r") as f:
        config = parse_config(f.read())
    with open(args.queries, "r") as f:
        queries = list(csv.reader(f))[1:]
    queries_text = [q[0] for q in queries]
    scaffold = PredictScaffold(config, args.model)
    queries = scaffold.tokenizer_doc.encode_batch(queries_text)
    encoded_doc = [
        scaffold.model.encoder_doc(
            t.tensor([q.ids], device=scaffold.device), t.ones(1, len(q), dtype=t.bool, device=scaffold.device)
        )
        for q in queries
    ]

    predictions = []
    h5_file = h5py.File("../cache/data/search.h5", "r")
    tokens_datasets = {
        lang: CodeDataset(h5_file["go"], "go", scaffold.device)
        for lang in ["go", "java", "javascript", "ruby", "python", "php"]
    }
    for data_file in args.data:
        index = LangIndex(config.model.encoded_dims, "angular")
        f = gzip.open(data_file, "rt")
        language: str = ""
        snippet = f.readline()
        i = 0
        while snippet:
            snippet = json.loads(snippet)
            language = snippet["language"]
            encoder = scaffold.model.encoders_code[language]
            _, (tokens, mask) = tokens_datasets[language][i]
            repr = encoder(tokens.unsqueeze(0), mask.unsqueeze(0))
            index.append(repr[0], snippet["identifier"], snippet["url"])
            snippet = f.readline()
            i += 1
        index.index.build(10)
        logging.info(f"Built index for {language}")

        for query_id, query_repr in enumerate(encoded_doc):
            query = queries_text[query_id]
            ranked_ids = index.index.get_nns_by_vector(query_repr[0], 100)
            for id in ranked_ids:
                func_name = index.func_names[id]
                url = index.urls[id]
                predictions.append((query, language, func_name, url))
        logging.info(f"Predictions complete for {language}")

    predictions = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
    predictions.to_csv(args.output, index=False)
    if enable_wandb:
        wandb.save(args.output)


if __name__ == "__main__":
    main()
