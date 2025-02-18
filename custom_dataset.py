"""
    Create a custom dataset class for training ResEmoteNet
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

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
        pixel_str = self.df.loc[index, "pixels"]
        pixel_val = np.fromstring(pixel_str, sep=" ", dtype=np.int8)
        pixel_arr = pixel_val.reshape(48, 48)
        
        # Convert numpy array into a PIL image in grayscale
        image = Image.fromarray(pixel_arr, mode="L")
        if self.transform:
            image = self.transform(image)
        
        label = self.df.loc[index, "emotion"]
        return image, label

