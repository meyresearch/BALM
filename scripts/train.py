import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

from balm import common_utils
from balm.configs import Configs
from balm.trainer import Trainer


def argument_parser():
    parser = argparse.ArgumentParser(description="BALM")
    parser.add_argument("--config_filepath", type=str, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to WandB
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    trainer = Trainer(configs, wandb_entity, wandb_project, outputs_dir)
    # train_ratio is only useful for Mpro, the rest will be null
    trainer.set_dataset(train_ratio=configs.dataset_configs.train_ratio)
    trainer.setup_training()
    trainer.train()


if __name__ == "__main__":
    main()
