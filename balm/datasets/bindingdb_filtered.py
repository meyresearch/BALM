from collections import defaultdict
from random import Random
from typing import Dict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def create_scaffold_split_dti(df, seed, frac, drug_column):
    """
    Create scaffold split for drug-target interaction data.
    It first generates molecular scaffold for each drug molecule
    and then split based on scaffolds while considering the drug-target pairs.

    Args:
        df (pd.DataFrame): Dataset dataframe with drug-target interactions.
        seed (int): The random seed.
        frac (list): A list of train/valid/test fractions, e.g., [0.7, 0.2, 0.1].
        drug_column (str): The column name where drug molecules (SMILES) are stored.

    Returns:
        dict: A dictionary of split dataframes with keys 'train', 'valid', and 'test'.
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


class BindingDBDataset:
    """
    A class to represent the BindingDB dataset for drug-target interactions.

    Attributes:
        filepath (str): The path to the BindingDB dataset CSV file.
        data (pd.DataFrame): The processed dataset.
        y (np.ndarray): The target values from the dataset.

    Methods:
        get_split(method, frac, seed, column_name): Returns the dataset split according to the specified method.
    """

    def __init__(self, filepath="data/BindingDB_filtered.csv"):
        """
        Initialize the BindingDBDataset object by loading and processing the dataset.

        Args:
            filepath (str): The path to the dataset file.
        """
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
    def _create_random_split(df, fold_seed, frac):
        """
        Create a random split of the dataset into train, validation, and test sets.

        Args:
            df (pd.DataFrame): The dataset to split.
            fold_seed (int): The random seed.
            frac (list): A list of train/valid/test fractions, e.g., [0.7, 0.2, 0.1].

        Returns:
            dict: A dictionary of split dataframes with keys 'train', 'valid', and 'test'.
        """
        _, val_frac, test_frac = frac
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
    
    @staticmethod
    def _create_fold_setting_cold(df, fold_seed, frac, entities):
        """
        Create a cold-split for the dataset, ensuring that specific entities
        (e.g., drugs or targets) are exclusive to one of the train, validation, or test sets.

        Args:
            df (pd.DataFrame): The dataset to split.
            fold_seed (int): The random seed.
            frac (list): A list of train/valid/test fractions, e.g., [0.7, 0.2, 0.1].
            entities (str or list): The column(s) to base the cold split on.

        Returns:
            dict: A dictionary of split dataframes with keys 'train', 'valid', and 'test'.
        """
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

    def get_split(
        self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Drug"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get a dataset split based on the specified method.

        Args:
            method (str): The split method ('random', 'scaffold', 'cold_drug', 'cold_target').
            frac (list): A list of train/valid/test fractions, e.g., [0.7, 0.2, 0.1].
            seed (int): The random seed.
            column_name (str): The column name to base the split on (used in scaffold split).

        Returns:
            dict: A dictionary of split dataframes with keys 'train', 'valid', and 'test'.
        """
        if method == "random":
            return self._create_random_split(self.data, seed, frac)
        elif method == "scaffold":
            return create_scaffold_split_dti(self.data, seed, frac, column_name)
        elif method == "cold_drug":
            return self._create_fold_setting_cold(self.data, seed, frac, entities="Drug")
        elif method == "cold_target":
            return self._create_fold_setting_cold(self.data, seed, frac, entities="Target")
        else:
            raise ValueError(f"Unknown split method: {method}")
