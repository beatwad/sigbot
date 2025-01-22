import os
from argparse import Namespace

import lightning as L
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

print(os.getcwd())

# constants
VAL_SIZE = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 4
LOAD_DIR = "data/signal_stat"
RANDOM_SEED = 28022024


class TabularDataset(Dataset):
    """Pytorch Dataset for tabular data storing"""

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class TabularDataModule(L.LightningDataModule):
    """LightningDataModule for tabular data processing and storing
    Parameters
    ----------
    batch_size
        Number of examples to operate on per forward step.
    val_size
        Validation data ratio in train / validation split.
    num_workers
        Number of additional processes to load data.
    load_dir
        Directory from which data will be loaded.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        val_size: float = VAL_SIZE,
        load_dir: str = LOAD_DIR,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.load_dir = load_dir

        self.input_shape = None
        self.output_dims = (1,)
        self.mapping = ["0", "1"]

        # prepare and split data
        self.setup()
        # get configuration of data
        self.data_config = self.config()
        print(self.data_config)

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_shape": self.input_shape,
            "output_dims": self.output_dims,
            "mapping": self.mapping,
            "batch_size": self.batch_size,
        }

    def setup(self, stage=None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.
        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        # load data
        df_buy = pd.read_pickle(f"{self.load_dir}/train_buy.pkl")
        df_sell = pd.read_pickle(f"{self.load_dir}/train_sell.pkl")
        df = pd.concat([df_buy, df_sell])
        df = df.sort_values("time").reset_index(drop=True)
        X = df.drop(columns=["time", "ticker", "pattern", "ttype"])
        y = df["target"]

        # split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        # standardize the numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # convert the NumPy arrays to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        self.input_shape = X_train.shape

        # create Pytorch DataSets for train and validation data
        self.data_train = TabularDataset(X_train, y_train)
        self.data_val = TabularDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.data_test,
    #         shuffle=False,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.on_gpu,
    #     )
