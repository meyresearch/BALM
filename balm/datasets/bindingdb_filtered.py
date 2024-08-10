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


class BindingDBDataset:
    def __init__(self, filepath="data/BindingDB_filtered.csv"):
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
        """Create random split."""
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

    def get_split(
        self, method="random", frac=[0.7, 0.2, 0.1], seed=42, column_name="Drug"
    ) -> Dict[str, pd.DataFrame]:
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