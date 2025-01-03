import numpy as np
import pandas as pd

from datasets import load_dataset

from .base_few_shot_dataset import FewShotDataset


class USP7Dataset(FewShotDataset):
    def __init__(self, train_ratio):
        super().__init__("USP7", train_ratio)