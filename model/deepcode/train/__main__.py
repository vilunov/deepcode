from . import Scaffold

from tqdm import tqdm


def main():
    scaffold = Scaffold()
    for epoch in range(100):
        with tqdm(desc="Training") as pbar:
            scaffold.epoch_train(pbar)
        with tqdm(desc="Validation") as pbar:
            mrr = scaffold.epoch_validate(pbar)
        print(mrr)
    scaffold.close()


if __name__ == "__main__":
    main()
