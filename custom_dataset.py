"""
    Create a custom dataset class for training ResEmoteNet
"""

from torch.utils.data import Dataset
import pandas as pd
from torch import Tensor
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
            csv_file: a file with each row consisting of emotion and pixels column
            tranform: an image data transformation pipeline
        """
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = str_to_tensor(self.df["pixels"][index])
        if self.transform:
            image = self.transform(image)
        
        label = self.df["emotion"][index]
        return image, label


def str_to_tensor(pixels:str) -> Tensor:
    """
        Convert pixel string into a normalized pytorch tensor
    """
    # String to numpy array
    p = np.fromstring(pixels, sep=" ", dtype=np.float32)

    # Feature normalization
    p_norm = p / 255.0

    # Reshape array to (C, H, W)
    p_reshaped = np.reshape(p_norm, (1, 48, 48))

    # Convert to tensor
    return torch.tensor(p_reshaped, dtype=torch.float32)
