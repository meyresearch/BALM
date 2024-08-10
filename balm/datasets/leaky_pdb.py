import numpy as np
import pandas as pd


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