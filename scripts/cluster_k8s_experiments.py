import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml
from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="Protein-Ligand Binding Affinity")
    parser.add_argument("--run_configs_filepath", type=str, required=True)
    parser.add_argument("--user_email", type=str, required=True)
    parser.add_argument("--git_branch", type=str, default="main")
    parser.add_argument("--namespace", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.run_configs_filepath, "r"))

    base_args = f"git clone https://$GIT_TOKEN@github.com/rgorantla04/DL_BA.git --branch {args.git_branch} && cd DL_BA && huggingface-cli download aryopg/balm_ext_datasets --repo-type dataset --local-dir data --token $HF_DOWNLOAD_TOKEN --quiet && pip install -U transformers && "

    name_mappings = {
        "bindingdb-cold-target": "bdb-ct",
        "bindingdb-cleaned-cold-target": "bdb-c-ct",
        "bindingdb-cleaned-cold-drug": "bdb-c-cd",
        "bindingdb-cold-drug": "bdb-cd",
        "bindingdb-scaffold": "bdb-s",
        "bindingdb-random": "bdb-r",
        "davis-cold-target": "d-ct",
        "davis-cold-drug": "d-cd",
        "davis-random": "d-r",
        "davis-scaffold": "d-s",
        "projection-tuning": "proj",
        "balanced-mse": "b-mse",
        "dropout": "do",
    }

    base_command = (
        "CUDA_LAUNCH_BLOCKING=1 accelerate launch scripts/train.py --config_filepath "
    )

    secret_env_vars = configs["env_vars"]
    commands = {}
    for config in configs["configs"]:
        run_name = "-".join(config["config_filepath"].split("/")[-2:])
        run_name = run_name.replace(".yaml", "").replace("_", "-")

        # Apply the mappings to the run_name
        for long_name, short_name in name_mappings.items():
            run_name = run_name.replace(long_name, short_name)

        # Remove 'tuning' from the run name if present
        run_name = run_name.replace("-tuning", "")

        commands[run_name] = {
            "command": base_command + config["config_filepath"],
            "gpu_product": config["gpu_product"],
            "gpu_limit": config["gpu_limit"],
        }

    for run_name, command in commands.items():
        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command['command']}")
        print(f"Job name: {run_name}")
        job = KubernetesJob(
            name=run_name,
            cpu_request="8",
            ram_request="64Gi",
            image=configs["image"],
            gpu_type="nvidia.com/gpu",
            gpu_limit=command["gpu_limit"],
            gpu_product=command["gpu_product"],
            backoff_limit=4,
            command=["/bin/bash", "-c", "--"],
            args=[base_args + command["command"]],
            secret_env_vars=secret_env_vars,
            user_email=args.user_email,
            namespace=args.namespace,
            kueue_queue_name=f"{args.namespace}-user-queue",
        )

        # Run the Job on the Kubernetes cluster
        job.run()


if __name__ == "__main__":
    main()
