import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as L
from torchmetrics.classification import BinaryAccuracy

class ListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def __len__(self):
        return len(self.list1)

    def __getitem__(self, idx):
        item1 = self.list1[idx]
        item2 = self.list2[idx]
        return item1, item2