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

# def compute_stats(data_file):
#     """
#         Compute mean and std of grayscale pixel values for stable training

#         data_file: name of a csv data file
#     """
#     # Scale down image pixel values to [0, 1]
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     ds = CustomDataset(data_file, transform)
#     dl = DataLoader(ds, batch_size=64, shuffle=False)

#     sum_pix = 0
#     sum_pix_squ = 0
#     num_pix = 0

#     for image, _ in dl:
#         sum_pix += image.sum()
#         sum_pix_squ += (image ** 2).sum()
#         num_pix += image.numel()

#     mean = sum_pix / num_pix
#     std = sqrt((sum_pix_squ / num_pix) - mean ** 2)

#     return mean.item(), std.item()

