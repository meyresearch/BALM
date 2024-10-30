# BALM: Binding Affinity Predictions with Protein and Ligand Language Models

**BALM** is a sequence-based deep learning framework for predicting **b**inding **a**ffinity using pretrained protein and ligand **l**anguage **m**odels.
BALM learns by optimizing the distance between protein and ligand embeddings in a shared space using the cosine similarity metric that directly represents experimental binding affinity.
We incorporate parameter-efficient fine-tuning methods in the BALM framework to adapt pretrained protein and ligand language models for binding prediction.

🧬 Tutorial notebook to get started: [`few_shot_demo.ipynb`](scripts/notebooks/few_shot_demo.ipynb)
📁 Huggingface Dataset: [`BALM-benchmark`](https://huggingface.co/datasets/BALM/BALM-benchmark)
🧠 Pretrained models: [`BALM models`](https://huggingface.co/BALM)

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
- HF_TOKEN  # Use your write token to give READ and WRITE privilege
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

### BindingDB with Random Split training

To train BALM on the BindingDB with random splits, you can run this config:

```bash
python scripts/train.py --config_filepath configs/random_seed_experiments/bindingdb_random/esm_lokr_chemberta_loha_cosinemse_1.yaml
```

In the paper, we reported the average metrics across multiple runs, and these individual runs are denoted by the suffix of the YAML file (e.g., `_1`, `_2`, or `_3`). The difference is only on the random seed value (e.g., `12`, `123`, `1234`).

### LeakyPDB with Random Split training

Similar to the BindingDB training, the LeakyPDB with Random Split training can be run using this config:

```bash
python scripts/train.py --config_filepath configs/random_seed_experiments/leakypdb/esm_lokr_chemberta_loha_cosinemse_1.yaml
```

## 💬 Feedback

Found a bug or wanted to suggest something? Please reach out via the [GitHub issues](https://github.com/meyresearch/BALM/issues).

## Citations

If you find the BALM model and benchmark useful in your research, please cite our paper:

```
TBA
```

## Licence 

See [LICENSE.md](LICENSE.md).

To discuss commercial use of our models, reach us [via email](mailto:antonia.mey@ed.ac.uk).

## Contact us

- Antonia Mey ([antonia.mey@ed.ac.uk](mailto:antonia.mey@ed.ac.uk))
