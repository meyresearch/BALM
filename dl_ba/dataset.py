import os
import random
from collections import defaultdict
from random import Random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class BindingDBDataset:
    def __init__(self, filepath="data1/BindingDB_cleaned.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        self.y = self.data["Y"].values
        self.data = (
            self.data.groupby(["Drug_ID", "Drug", "Target_ID", "Target"])
            .Y.agg(max)
            .reset_index()
        )

    def get_split(
        self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Drug"
    ) -> Dict[str, pd.DataFrame]:
        if method == "random":
            return create_fold(self.data, seed, frac)
        elif method == "scaffold":
            return create_scaffold_split_dti(self.data, seed, frac, column_name)
        elif method == "cold_drug":
            return create_fold_setting_cold(self.data, seed, frac, entities="Drug")
        elif method == "cold_target":
            return create_fold_setting_cold(self.data, seed, frac, entities="Target")
        else:
            raise ValueError(f"Unknown split method: {method}")


class RandomTargetBindingDBDataset:
    def __init__(self, filepath="data1/BindingDB_cleaned.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        self.y = self.data["Y"].values

        self.data = (
            self.data.groupby(["Drug_ID", "Drug", "Target_ID", "Target"])
            .Y.agg(max)
            .reset_index()
        )

    @staticmethod
    def _create_random_target_sequence(seq_length, seed):
        random.seed(seed)
        return "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=seq_length))

    def _randomise_target(self, data, seed):
        original_targets = data["Target"]

        randomised_target = data["Target"].apply(
            lambda x: self._create_random_target_sequence(len(x), seed)
        )
        print(randomised_target)

        assert not original_targets.equals(randomised_target)

        return randomised_target

    def get_split(
        self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Target"
    ) -> Dict[str, pd.DataFrame]:
        if method == "random":
            data_split = create_fold(self.data, seed, frac)
            data_split["train"]["Target"] = self._randomise_target(
                data_split["train"], seed
            )
            return data_split


def read_ligands_from_dataset(filepath):
    data = pd.read_csv(filepath)
    unique_ligands = data['Drug'].unique()
    return list(unique_ligands)


# class RandomDrugBindingDBDataset:
#     def __init__(self, filepath="data1/BindingDB_cleaned.csv"):
#         self.filepath = filepath
#         self.data = pd.read_csv(filepath)
#         self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
#         self.y = self.data["Y"].values

#         self.data = (
#             self.data.groupby(["Drug_ID", "Drug", "Target_ID", "Target"])
#             .Y.agg(max)
#             .reset_index()
#         )

#     def _randomise_drug(self, data, seed):
#         original_drugs = data["Drug"]
#         randomised_drug = data["Drug"].sample(frac=1, random_state=seed).values

#         assert not original_drugs.equals(randomised_drug)

#         return randomised_drug

    # def get_split(
    #     self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Drug"
    # ) -> Dict[str, pd.DataFrame]:
    #     if method == "random":
    #         data_split = create_fold(self.data, seed, frac)
    #         data_split["train"]["Drug"] = self._randomise_drug(
    #             data_split["train"], seed
    #         )
    #         return data_split

class RandomDrugBindingDBDataset:
    def __init__(self, filepath="data1/BindingDB_cleaned.csv", ligands_filepath=None):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        self.y = self.data["Y"].values

        self.data = (
            self.data.groupby(["Drug_ID", "Drug", "Target_ID", "Target"])
            .Y.agg(max)
            .reset_index()
        )
        
        self.ligands_list = read_ligands_from_dataset(filepath)

    def _randomise_drug(self, data, seed):
        random.seed(seed)
        randomised_drugs = [random.choice(self.ligands_list) for _ in range(len(data))]
        return randomised_drugs
    
    def get_split(
        self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Drug"
    ) -> Dict[str, pd.DataFrame]:
        if method == "random":
            data_split = create_fold(self.data, seed, frac)
            data_split["train"]["Drug"] = self._randomise_drug(
                data_split["train"], seed
            )
            return data_split





def create_fold(df, fold_seed, frac):
    """Create random split."""
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """Create cold-split where given one or multiple columns, it first splits based on entities in the columns and then maps all associated data points to the partition."""
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_scaffold_split_dti(df, seed, frac, drug_column):
    """
    Create scaffold split for drug-target interaction data.
    It first generates molecular scaffold for each drug molecule
    and then split based on scaffolds while considering the drug-target pairs.

    Args:
        df (pd.DataFrame): dataset dataframe with drug-target interactions
        seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        drug_column (str): the column name where drug molecules (SMILES) are stored

    Returns:
        dict: a dictionary of split dataframes (train/valid/test)
    """

    random = Random(seed)

    # Generate scaffolds for each drug
    scaffolds = defaultdict(set)
    for idx, row in df.iterrows():
        smiles = row[drug_column]
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds[scaffold].add(idx)
        except:
            continue

    # Split scaffolds into train, valid, test sets
    scaffolds = list(scaffolds.values())
    random.shuffle(scaffolds)

    train_size = int(len(df) * frac[0])
    valid_size = int(len(df) * frac[1])

    train, valid, test = set(), set(), set()
    for scaffold_set in scaffolds:
        if len(train) + len(scaffold_set) <= train_size:
            train.update(scaffold_set)
        elif len(valid) + len(scaffold_set) <= valid_size:
            valid.update(scaffold_set)
        else:
            test.update(scaffold_set)

    # Create DataFrame subsets for each split
    train_df = df.iloc[list(train)].reset_index(drop=True)
    valid_df = df.iloc[list(valid)].reset_index(drop=True)
    test_df = df.iloc[list(test)].reset_index(drop=True)

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


class LeakyPDBDataset:
    def __init__(self, filepath="data/leaky_pdb.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        # Rename columns for consistency with TDC data
        # smiles -> Drug
        # seq -> Target
        self.data = self.data.rename(
            columns={"smiles": "Drug", "seq": "Target", "value": "Y"}
        )

        # for dGs
        # self.data = self.data.rename(
        #     columns={"smiles": "Drug", "seq": "Target"}
        # )

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")
        self.y = self.data["Y"].values

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        # Splitting data based on "new_split" values with additional conditions CL1 for train, CL2 for valid/test
        train = self.data[
            (self.data["new_split"] == "train")
            & self.data["CL1"]
            & ~self.data["covalent"]
        ].reset_index(drop=True)
        valid = self.data[
            (self.data["new_split"] == "val")
            & self.data["CL2"]
            & ~self.data["covalent"]
        ].reset_index(drop=True)
        test = self.data[
            (self.data["new_split"] == "test")
            & self.data["CL2"]
            & ~self.data["covalent"]
        ].reset_index(drop=True)

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class MproDataset:
    def __init__(self, train_ratio, filepath="data/Mpro_data.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        self.y = self.data["Y"].values

        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    # def get_split(self, *args, **kwargs):
    #     """Create data splits based on 'new_split' column.

    #     Args:
    #         df (pd.DataFrame): The dataset DataFrame.

    #     Returns:
    #         dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
    #     """
    #     if self.train_ratio > 0:
    #         # Separating data randomly
    #         ids = range(len(self.data))
    #         ids = np.random.permutation(ids)
    #         train_ids = ids[: int(self.train_ratio * len(ids))]
    #         test_ids = ids[int(self.train_ratio * len(ids)) :]

    #         train = self.data.iloc[train_ids].reset_index(drop=True)
    #         test = self.data.iloc[test_ids].reset_index(drop=True)
    #     else:
    #         train = None
    #         test = self.data

    #     return {
    #         "train": train,
    #         "valid": None,
    #         "test": test,
    #     }

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.1 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = self.data.iloc[val_ids].reset_index(drop=True)
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class USP7Dataset:
    def __init__(self, train_ratio, filepath="data1/USP7_filtered.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        self.y = self.data["Y"].values

        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.1 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = self.data.iloc[val_ids].reset_index(drop=True)
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class HSP9Dataset:
    def __init__(self, train_ratio, filepath="data/HSP9_data.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        self.y = self.data["Y"].values
        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.0 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = (
                self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
            )
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class KITDataset:
    def __init__(self, train_ratio, filepath="data1/KIT_data_reg.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        # self.y = self.data["Y"].values
        self.y = self.data["Y"].astype(float).values

        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.0 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = (
                self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
            )
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


# class KinasesDataset:
#     def __init__(self, train_ratio, filepath="data1/Kinases_data.csv"):
#         self.filepath = filepath
#         self.data = pd.read_csv(filepath)

#         #self.y = self.data["Y"].values
#         self.y = self.data["Y"].astype(float).values

#         self.train_ratio = train_ratio

#         print(f"Total of original data: {len(self.data)}")
#         # Remove rows where smiles is NaN
#         self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
#         print(f"Total after filtering: {len(self.data)}")

#     def get_split(self, *args, **kwargs):
#         """Create data splits based on 'new_split' column.

#         Args:
#             df (pd.DataFrame): The dataset DataFrame.

#         Returns:
#             dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
#         """
#         if self.train_ratio > 0:
#             # Separating data randomly
#             ids = np.arange(len(self.data))
#             np.random.shuffle(ids)

#             # Define the size of each split
#             val_size = int(0.0 * len(ids))  # 10% for validation
#             train_size = int(
#                 self.train_ratio * (len(ids) - val_size)
#             )  # Remaining for train

#             # Split the ids
#             val_ids = ids[:val_size]
#             remaining_ids = ids[val_size:]
#             train_ids = remaining_ids[:train_size]
#             test_ids = remaining_ids[train_size:]

#             # Create the splits
#             train = self.data.iloc[train_ids].reset_index(drop=True)
#             valid = (
#                 self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
#             )
#             test = self.data.iloc[test_ids].reset_index(drop=True)
#         else:
#             train = None
#             valid = None
#             test = self.data

#         return {
#             "train": train,
#             "valid": valid,
#             "test": test,
#         }


class KinasesDataset:
    def __init__(self, train_ratio, filepath="data1/Kinases_data.csv"):
        """
        Initialize the KinasesDataset class.

        Args:
            train_ratio (float): Ratio of data to be used for training.
            filepath (str): Path to the CSV file containing the data.
        """
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        # Convert "Y" column to float values and remove rows with NaN in "Y"
        self.data["Y"] = self.data["Y"].astype(float)
        self.data = self.data.dropna(subset=["Y"])

        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")

        # Remove rows where "Drug" is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    def get_split(self, *args, **kwargs):
        """
        Create data splits based on a specified ratio for training and testing.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.1 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for training

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = (
                self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
            )
            test = self.data.iloc[test_ids].reset_index(drop=True)

            print(f"Train size: {len(train)}")
            print(f"Validation size: {len(valid)}")
            print(f"Test size: {len(test)}")

            return {
                "train": train,
                "valid": valid,
                "test": test,
            }
        else:
            # If train_ratio is 0 or less, return the whole data as the test set
            print(f"Using the entire dataset for testing. Size: {len(self.data)}")
            return {
                "train": None,
                "valid": None,
                "test": self.data,
            }


class CATSDataset:
    def __init__(self, train_ratio, filepath="data/CATS_data.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        self.y = self.data["Y"].values
        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")
        print(f"All Y values set to: {self.data['Y'].unique()}")  # Debug statement

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.0 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = (
                self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
            )
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class CASPDataset:
    def __init__(self, train_ratio, filepath="data1/CASP16_CathepsinG_data.csv"):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

        self.y = self.data["Y"].values
        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows where smiles is NaN
        self.data = self.data.dropna(subset=["Drug"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")
        print(f"All Y values set to: {self.data['Y'].unique()}")  # Debug statement

    def get_split(self, *args, **kwargs):
        """Create data splits based on 'new_split' column.

        Args:
            df (pd.DataFrame): The dataset DataFrame.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.0 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = (
                self.data.iloc[val_ids].reset_index(drop=True) if len(val_ids) else None
            )
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }


class DataCollatorWithPadding:
    def __init__(
        self,
        protein_tokenizer: PreTrainedTokenizerBase,
        drug_tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        protein_max_length: Optional[int] = None,
        drug_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.protein_tokenizer = protein_tokenizer
        self.drug_tokenizer = drug_tokenizer
        self.padding = padding
        self.protein_max_length = protein_max_length
        self.drug_max_length = drug_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract 'protein_input_ids' and prepare them for padding
        protein_features = [
            {"input_ids": feature["protein_input_ids"]} for feature in features
        ]

        # Pad 'protein_input_ids' and ensure they're named correctly after padding
        padded_protein_features = self.protein_tokenizer.pad(
            protein_features,
            padding=self.padding,
            max_length=self.protein_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Extract 'drug_input_ids' and prepare them for padding
        drug_features = [
            {"input_ids": feature["drug_input_ids"]} for feature in features
        ]

        # Pad 'drug_input_ids' and ensure they're named correctly after padding
        padded_drug_features = self.drug_tokenizer.pad(
            drug_features,
            padding=self.padding,
            max_length=self.drug_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "protein_ori_sequences": [
                feature["protein_ori_sequences"] for feature in features
            ],
            "drug_ori_sequences": [
                feature["drug_ori_sequences"] for feature in features
            ],
            "protein_input_ids": padded_protein_features["input_ids"],
            "protein_attention_mask": padded_protein_features["attention_mask"],
            "drug_input_ids": padded_drug_features["input_ids"],
            "drug_attention_mask": padded_drug_features["attention_mask"],
            "labels": torch.stack([torch.tensor(feature["Y"]) for feature in features]),
        }

        return batch
