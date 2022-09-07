import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class StockDataset(Dataset):
    def __init__(self, stock_name: str, input_window_size: int, output_window_size: int, transform = None):
        self.data_path = './dataset/nasdaq100_padding.csv'
        self.stock_name = stock_name
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.total_window_size = input_window_size + output_window_size
        self.transform = transform
        self.dataset = self._load_data()

    def __len__(self):
        return self.dataset.shape[0] - self.total_window_size + 1

    def __getitem__(self, index):
        if isinstance(index, slice) :
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*index.indices(len(self)))]
        
        X = self.dataset[index: index+self.input_window_size]
        y = self.dataset[index+self.input_window_size: index+self.total_window_size].reshape((1))
        if self.transform:
            # X, y = self.transform(X), self.transform(y)
            X, y = torch.from_numpy(X).type(torch.Tensor), torch.from_numpy(y).type(torch.Tensor)
        return X, y

    def _load_data(self):
        return np.array(pd.read_csv(self.data_path, usecols=[self.stock_name]), dtype=np.float32)

class StockDatasetModule(LightningDataModule):
    def __init__(self, stock_name: str, input_window_size: int, output_window_size: int, batch_size = 32, transform = None):
        super().__init__()
        self.stock_name = stock_name
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        dataset = StockDataset(self.stock_name, self.input_window_size, self.output_window_size)

        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        # self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.train_dataset, self.test_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, train_size + test_size))

        train_size = int(0.7 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        # self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])
        self.train_dataset, self.val_dataset = Subset(self.train_dataset, range(train_size)), Subset(self.train_dataset, range(train_size, train_size + val_size))

        self.train_dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset)