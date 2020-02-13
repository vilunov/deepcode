import json

import torch
from tqdm import tqdm

from deepcode.train import Scaffold


def main():
    scaffold = Scaffold()
    torch.save(scaffold.model.state_dict(), f"../cache/models/start.pickle")
    for epoch in range(100):
        scaffold.epoch_train(tqdm)
        metrics = scaffold.epoch_validate(tqdm)
        torch.save(scaffold.model.state_dict(), f"../cache/models/{epoch}.pickle")
        with open(f"../cache/models/{epoch}.json", "w") as file:
            json.dump(metrics, file)
        print(metrics)
    scaffold.close()


if __name__ == "__main__":
    main()
