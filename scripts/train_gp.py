import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import wandb
import pandas as pd
import numpy as np
import torch
import gpytorch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Train Gaussian Process models on ECFP8 and ChemBERTa embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/Mpro.csv",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["ECFP8", "ChemBERTa"],
        help="Type of embedding on which the GP is trained on",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for data splitting."
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training iterations."
    )

    return parser.parse_args()


def smiles_to_ecfp8_fingerprint(smiles_list):
    """Convert SMILES to ECFP8 Morgan Fingerprints."""
    fingerprints = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fingerprints.append(
                morgan_gen.GetFingerprint(mol)
            )  # ECFP8
        else:
            fingerprints.append(np.zeros(2048))  # Handle invalid molecules
    return np.array(fingerprints)


def smiles_to_chemberta_embedding(smiles_list, tokenizer, model):
    """Convert SMILES strings into ChemBERTa embeddings."""
    embeddings = []
    for smi in smiles_list:
        inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)


def split_data(X, y, test_size=0.8, seed=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


class ExactGPModelECFP8(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelECFP8, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelChemBERTa2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelChemBERTa2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    X_train, y_train, X_test, y_test, model_class, learning_rate=0.1, epochs=50
):
    """Train GP model and log to Weights & Biases."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_class(X_train, y_train, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize training logging for W&B
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        # Log the loss to W&B
        wandb.log({"epoch": i, "train/loss": loss.item()})

    # Evaluation mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test)).mean

    preds = preds.detach().numpy()
    y_test = y_test.detach().numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    spearman_corr, _ = spearmanr(y_test, preds)

    # Log evaluation metrics to W&B
    wandb.log({"test/rmse": rmse, "test/r2": r2, "test/spearman": spearman_corr})

    # Save predictions and actual values to a CSV
    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": preds})
    predictions_filename = "test_prediction.csv"
    predictions_df.to_csv(predictions_filename, index=False)

    # Create W&B artifact for the predictions
    artifact = wandb.Artifact("test_prediction", type="prediction")
    artifact.add_file(predictions_filename)
    wandb.log_artifact(artifact)

    return rmse, r2, spearman_corr


def main():
    args = argument_parser()
    # Load dataset
    data = pd.read_csv(args.dataset)

    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    dataset_name = os.path.basename(args.dataset).split(".")[0]

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"gp_{args.embedding_type}_{args.lr}_{dataset_name}_{args.random_seed}",
        config=args,
    )

    # Extract SMILES and target (Y)
    smiles = data["Drug"]
    y = data["Y"]

    # Standardize target values (Y)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    if args.embedding_type == "ECFP8":
        # Convert SMILES to ECFP8 fingerprints
        X_ecfp8 = smiles_to_ecfp8_fingerprint(smiles)

        # Split data
        X_train_ecfp8, X_test_ecfp8, y_train, y_test = split_data(
            X_ecfp8, y_scaled, seed=args.random_seed
        )
        # Train and evaluate ECFP8-based GP
        rmse_ecfp8, r2_ecfp8, spearman_ecfp8 = train_gp_model(
            X_train_ecfp8,
            y_train,
            X_test_ecfp8,
            y_test,
            ExactGPModelECFP8,
            args.lr,
            args.epochs,
        )
        # Print results
        print(
            f"ECFP8 GP Model - RMSE: {rmse_ecfp8}, "
            f"R^2: {r2_ecfp8}, "
            f"Spearman: {spearman_ecfp8}"
        )
    elif args.embedding_type == "ChemBERTa":
        # Load ChemBERTa tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

        # Convert SMILES into ChemBERTa embeddings
        X_chemberta2 = smiles_to_chemberta_embedding(smiles, tokenizer, model)

        X_train_chemberta2, X_test_chemberta2, y_train, y_test = split_data(
            X_chemberta2, y_scaled, seed=args.random_seed
        )

        # Train and evaluate ChemBERTa2-based GP
        rmse_chemberta2, r2_chemberta2, spearman_chemberta2 = train_gp_model(
            X_train_chemberta2,
            y_train,
            X_test_chemberta2,
            y_test,
            ExactGPModelChemBERTa2,
            args.lr,
            args.epochs,
        )
        # Print results
        print(
            f"ChemBERTa2 GP Model - RMSE: {rmse_chemberta2}, "
            f"R^2: {r2_chemberta2}, "
            f"Spearman: {spearman_chemberta2}"
        )

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
