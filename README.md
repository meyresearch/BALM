# BALM: Binding Affinity Predictions with Protein and Ligand Language Models

**BALM** is a sequence-based deep learning framework for predicting **b**inding **a**ffinity using pretrained protein and ligand **l**anguage **m**odels.
BALM learns by optimizing the distance between protein and ligand embeddings in a shared space using the cosine similarity metric that directly represents experimental binding affinity.
We incorporate parameter-efficient fine-tuning methods in the BALM framework to adapt pretrained protein and ligand language models for binding prediction.

🧬 Tutorial notebook to get started: [`few_shot_demo.ipynb`](scripts/notebooks/few_shot_demo.ipynb)

## Setup

### Create a conda environment

```bash
conda env create -f environment.yaml
conda activate balm
```

### Create an environment file

To download from/upload to HF hub, create a `.env` file containing
```
- WANDB_ENTITY  # For experiment logging to WandB
- WANDB_PROJECT_NAME  # For experiment logging to WandB
- HF_TOKEN  # Use your write token to give READ and WRITE priviledge
```

Check `.env.example`. You can use it to create your own `.env` file.

### Dataset Access

We published our dataset via HuggingFace: https://huggingface.co/datasets/BALM/BALM-benchmark.
The code will automatically download the data from this link accordingly.
Check the dataset page to know more details about it!

## Training

```bash
python scripts/train.py --config_filepath path/to/config_file.yaml
```

You can find config files in the [`configs`](configs/) folder.
