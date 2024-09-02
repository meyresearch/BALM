import numpy as np
import pandas as pd

from datasets import load_dataset


class CATSDataset:
    def __init__(self, train_ratio, filepath="data/CATS_data.csv"):
        self.data = load_dataset("BALM/BALM-benchmark", "CATS", split="train").to_pandas()

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