#  python main.py --config config/ResNet.yaml --wandb 'disabled'
#  python main.py --config config/ResNet.yaml --wandb 'online'
import argparse
import functools
import os

import wandb
import yaml

from train import train


def main():
    parser = argparse.ArgumentParser(description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default='baseline')
    parser.add_argument("--config", type=str, help="Config file", default='config/ResNet.yaml')
    parser.add_argument("--wandb", type=str, help="WandB mode", default='online')
    parser.add_argument(
        "--dataset_path", type=str, help="Dataset path", default='/ghome/group03/mcv/m3/datasets/MIT_small_train_1'
    )
    args = parser.parse_args()

    # get the path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(path, args.config)

    with open(path_config) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="M5_W1")
    wandb.agent(sweep_id, function=functools.partial(train, args))


if __name__ == "__main__":
    main()
    wandb.finish()
