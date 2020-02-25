import os
import json
import argparse
from datetime import datetime
import logging

import torch
from torch.multiprocessing import set_start_method
from tqdm import tqdm

from deepcode.train import Scaffold


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", dest="weights_path", type=str, required=False)
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, default=1e-1)
    parser.add_argument("--loss", dest="loss", type=str, default="crossentropy")
    parser.add_argument("--loss-margin", dest="loss_margin", required=False, type=float)
    parser.add_argument("-do", "--dropout", dest="dropout", default=0.3, type=float)
    parser.add_argument("-rs", "--representation-size", dest="repr_size", default=128, type=int)
    parser.add_argument("-bs", "--batch-size", dest="batch_size", default=512, type=int)
    args = parser.parse_args()
    print(args)
    return args


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO)
    scaffold = Scaffold(arguments())
    save_path = "../cache/models/" + datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S")
    logging.info("Starting app")
    os.makedirs(save_path)
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
