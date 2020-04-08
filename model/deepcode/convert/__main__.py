import argparse
import pickle
import gzip
import json

from tqdm import tqdm

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, dest="input", help="input file")
    parser.add_argument("-o", "--output", type=str, required=True, dest="output", help="output file")
    args = parser.parse_args()
    print(args)
    return args


def main():
    retain_keys = {"url", "identifier", "language", "function_tokens"}
    args = arguments()
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    print(len(data))
    with gzip.open(args.output, "wt") as f:
        for line in tqdm(data):
            for k in set(line.keys()):
                if k not in retain_keys:
                    del line[k]
            line_s = json.dumps(line)
            f.write(line_s)
            f.write("\n")


if __name__ == '__main__':
    main()
