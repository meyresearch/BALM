import numpy as np
import pandas as pd

from datasets import load_dataset

from .base_few_shot_dataset import FewShotDataset


class SYKDataset(FewShotDataset):
    def __init__(self, train_ratio):
        super().__init__("SYK", train_ratio)