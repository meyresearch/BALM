import random
from collections import defaultdict
from random import Random

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset


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


class PairedBindingDataset(Dataset):
    def __init__(
        self,
        data_df,
        protein_tokenized_dict,
        drug_tokenized_dict,
        num_negative_samples: int,
        query_entity_column: str = "Target",
    ):
        """_summary_

        Args:
            data_df (_type_): _description_
            anchor_entity (str): Either "Target" or "Drug"
        """
        super(PairedBindingDataset, self).__init__()
        self.data_df = data_df

        self.protein_tokenized_dict = protein_tokenized_dict
        self.drug_tokenized_dict = drug_tokenized_dict

        self.query_entity_column = query_entity_column
        self.value_entity_column = "Target" if query_entity_column == "Drug" else "Drug"

        self.num_negative_samples = num_negative_samples

        self.queries = self.data_df[self.data_df["Y"] == 1][
            self.query_entity_column
        ].unique()
        self.set_paired_samples_dict()

        self.paired_queries = list(self.paired_dataset.keys())

    def set_paired_samples_dict(self):
        self.paired_dataset = {}
        for query in self.queries:
            positives = self.data_df[
                (self.data_df[self.query_entity_column] == query)
                & (self.data_df["Y"] == 1)
            ][self.value_entity_column].to_list()
            negatives = self.data_df[
                (self.data_df[self.query_entity_column] == query)
                & (self.data_df["Y"] == 0)
            ][self.value_entity_column].to_list()

            if len(positives) > 0 and len(negatives) > 0:
                self.paired_dataset[query] = {
                    "positives": positives,
                    "negatives": negatives,
                }

    def __len__(self):
        return len(self.paired_queries)

    def __getitem__(self, idx):
        query = self.paired_queries[idx]

        if self.query_entity_column == "Target":
            query_inputs = self.protein_tokenized_dict[query]
        elif self.query_entity_column == "Drug":
            query_inputs = self.drug_tokenized_dict[query]

        # Sample 1 positive pair
        positive_sample = random.choice(self.paired_dataset[query]["positives"])
        if self.value_entity_column == "Target":
            positive_inputs = self.protein_tokenized_dict[positive_sample]
        elif self.value_entity_column == "Drug":
            positive_inputs = self.drug_tokenized_dict[positive_sample]

        # Sample n negative pairs
        negative_input_ids = []
        negative_attention_mask = []
        negative_samples = random.sample(
            self.paired_dataset[query]["negatives"],
            len(self.paired_dataset[query]["negatives"]),
        )[: self.num_negative_samples]

        for value in negative_samples:
            if self.value_entity_column == "Target":
                value_inputs = self.protein_tokenized_dict[value]
            elif self.value_entity_column == "Drug":
                value_inputs = self.drug_tokenized_dict[value]
            negative_input_ids += [value_inputs["input_ids"]]
            negative_attention_mask += [value_inputs["attention_mask"]]
        negative_inputs = {}
        negative_inputs["input_ids"] = negative_input_ids
        negative_inputs["attention_mask"] = negative_attention_mask

        return {
            "query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"],
            "positive_input_ids": positive_inputs["input_ids"],
            "positive_attention_mask": positive_inputs["attention_mask"],
            "negatives_input_ids": negative_inputs["input_ids"],
            "negatives_attention_mask": negative_inputs["attention_mask"],
        }
