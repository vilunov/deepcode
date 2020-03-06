import argparse
import csv
import gzip
import json

import torch as t

from deepcode.config import parse_config
from deepcode.predict.scaffold import PredictScaffold


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--queries", type=str, required=True, dest="queries", help="input queries file")
    parser.add_argument("-d", "--data", type=str, required=True, dest="data", help="input data files", nargs='+')
    parser.add_argument("-o", "--output", type=str, required=True, dest="output", help="output predictions file")
    parser.add_argument("-m", "--model", type=str, required=True, dest="model", help="model weights file")
    parser.add_argument("-c", "--config", type=str, required=True, dest="config", help="model configuration file")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = arguments()
    with open(args.config, "r") as f:
        config = parse_config(f.read())
    with open(args.queries, "r") as f:
        queries = list(csv.reader(f))[1:]
    queries_text = [q[0] for q in queries]
    scaffold = PredictScaffold(config, args.model)
    queries = scaffold.tokenizer_doc.encode_batch(queries_text)
    encoded_doc = [scaffold.model.encoder_doc(t.tensor([q.ids]), t.ones(1, len(q), dtype=t.bool)) for q in queries]
    encoded_doc = t.cat(encoded_doc)
    output_file = open(args.output, "w")
    output_file.write("query,language,identifier,url\n")
    for data_file in args.data:
        encoded_code = []
        with gzip.open(data_file, "r") as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        for i in range(len(data)):
            snippet = data[i]
            language = snippet["language"]
            tokenizer = scaffold.tokenizers_code[language]
            encoder = scaffold.model.encoders_code[language]
            tokens = snippet["code_tokens"]
            tokens = tokenizer.encode_batch(tokens)
            tokens = t.cat([t.tensor(i.ids, dtype=t.int64) for i in tokens])
            encoded_code.append(encoder(tokens.unsqueeze(0), t.ones(1, tokens.shape[0], dtype=t.bool)))
        encoded_code = t.cat(encoded_code)
        distances = t.matmul(encoded_code, encoded_doc.T)
        ranked = distances.argsort(dim=0)
        for query_id, ranked_query in enumerate(ranked.T):
            query = queries_text[query_id]
            for id in ranked_query:
                snippet = data[id]
                line = f'{query},{snippet["language"]},{snippet["func_name"]},{snippet["url"]}\n'
                output_file.write(line)
    output_file.close()


if __name__ == '__main__':
    main()
