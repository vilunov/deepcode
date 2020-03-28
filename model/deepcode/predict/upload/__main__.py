import wandb


def main():
    wandb.init(name="test")
    wandb.save("model_predictions.csv")


if __name__ == "__main__":
    main()
