import argparse
import csv
import logging
import pickle
import os
import glob
import gc
from collections import defaultdict
import multiprocessing

import parmap
import torch as t
import pandas as pd
from annoy import AnnoyIndex
from tokenizers import ByteLevelBPETokenizer

from deepcode.config import parse_config
from deepcode.predict.scaffold import PredictScaffold

tokenizers_code = {
    code: ByteLevelBPETokenizer(f"../cache/vocabs/code-{code}-vocab.json", f"../cache/vocabs/code-{code}-merges.txt")
    for code in ["java", "javascript", "php", "ruby", "python", "go"]
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--queries", type=str, required=True, dest="queries", help="input queries file")
    parser.add_argument("-d", "--data", type=str, required=True, dest="data", help="input data files", nargs="+")
    parser.add_argument("-m", "--models", type=str, required=True, dest="models", help="path to cache/models dir")
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


def process(snippet, device):
    tokenizer = tokenizers_code[snippet["language"]]
    tokens = snippet["function_tokens"]
    tokens = tokenizer.encode_batch(tokens)
    tokens = t.cat([t.tensor(i.ids, dtype=t.int64, device=device) for i in tokens])
    return (tokens, snippet["identifier"], snippet["url"])


def main():
    pool = multiprocessing.Pool()
    logging.basicConfig(level=logging.INFO)
    args = arguments()
    with open(args.queries, "r") as f:
        queries = list(csv.reader(f))[1:]
    queries_text = [q[0] for q in queries]

    model_groups = glob.glob(os.path.join(args.models, "*/"))

    # PREFETCH AND TOKENIZE DATA
    model_group = model_groups[0]
    config_path = os.path.join(model_group, "config.toml")
    with open(config_path, "r") as f:
        config = parse_config(f.read())
    model = glob.glob(os.path.join(model_group, "*.pickle"))[0]
    scaffold = PredictScaffold(config, model)

    queries = scaffold.tokenizer_doc.encode_batch(queries_text)
    max_len = max(len(q) for q in queries)
    queries_ids = t.stack(
        [
            t.cat(
                [
                    t.tensor(q.ids, device=scaffold.device),
                    t.zeros(max_len - len(q.ids), device=scaffold.device, dtype=t.long),
                ]
            )
            for q in queries
        ]
    )
    queries_mask = t.zeros(len(queries), max(len(q) for q in queries), dtype=t.bool, device=scaffold.device)
    for i, q in enumerate(queries):
        queries_mask[i, : len(q)] = True
    del queries

    logging.info(f"Loaded all queries {queries_ids.shape}")
    data_new = defaultdict(list)
    for data_file in args.data:
        logging.info(f"Loading from {data_file}")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Data loaded from {data_file}, len = {len(data)}")
        language = data[0]["language"]
        processed = parmap.map(process, data, scaffold.device, pm_pool=pool, pm_pbar=True)
        del data
        data_new[language] = processed
        logging.info(f"Processed {data_file}")
        gc.collect()
    logging.info("Loaded all data")

    # PROCESS ALL
    for model_group in model_groups:
        logging.info(f"Now processing dir {model_group}")
        config_path = os.path.join(model_group, "config.toml")
        with open(config_path, "r") as f:
            config = parse_config(f.read())
        models = glob.glob(os.path.join(model_group, "*.pickle"))
        for model in models:
            logging.info(f"Now processing model {model}")
            scaffold = PredictScaffold(config, model)
            encoded_doc = scaffold.model.encoder_doc(queries_ids, queries_mask)

            predictions = []
            for language, data in data_new.items():
                index = LangIndex(config.model.encoded_dims, "angular")
                for i, (tokens, identifier, url) in enumerate(data):
                    encoder = scaffold.model.encoders_code[language]
                    repr = encoder(
                        tokens.unsqueeze(0), t.ones(1, tokens.shape[0], dtype=t.bool, device=scaffold.device)
                    )
                    index.append(repr[0], identifier, url)
                index.index.build(10)
                logging.info(f"Built index for {language}")

                for query_id, query_repr in enumerate(encoded_doc):
                    query = queries_text[query_id]
                    ranked_ids = index.index.get_nns_by_vector(query_repr, 100)
                    for id in ranked_ids:
                        func_name = index.func_names[id]
                        url = index.urls[id]
                        predictions.append((query, language, func_name, url))
                logging.info(f"Predictions complete for {language}")

            predictions = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
            output = model.rsplit(".", 1)[0] + ".csv"
            predictions.to_csv(output, index=False)
            logging.info(f"Written csv for {model} to {output}")


if __name__ == "__main__":
    main()
