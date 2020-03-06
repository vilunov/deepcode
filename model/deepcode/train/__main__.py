import os
import json
import argparse
from datetime import datetime
import logging
from shutil import copyfile

import torch
from tqdm import tqdm

from deepcode.train import TrainScaffold
from deepcode.config import parse_config


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", dest="weights_path", type=str, required=False)
    parser.add_argument("-c", "--config", dest="config", type=str, required=True)
    args = parser.parse_args()
    print(args)
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = arguments()
    with open(args.config, "r") as f:
        config = parse_config(f.read())
    scaffold = TrainScaffold(config, args.weights_path)
    save_path = os.path.join("..", "cache", "models", datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S"))
    logging.info("Starting app")
    os.makedirs(save_path)
    copyfile(args.config, os.path.join(save_path, "config.toml"))
    for epoch in range(100):
        logging.info("Starting epoch")
        scaffold.epoch_train(tqdm)
        logging.info("Starting validation")
        metrics = scaffold.epoch_validate(tqdm)
        torch.save(scaffold.model.state_dict(), os.path.join(save_path, f"{epoch}.pickle"))
        with open(os.path.join(save_path, f"{epoch}.json"), "w") as file:
            json.dump(metrics, file)
        logging.info(f"Validation results: {metrics}")
    scaffold.close()


if __name__ == "__main__":
    main()
