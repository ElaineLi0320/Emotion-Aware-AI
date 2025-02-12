"""
    Prepare data for model training
    
    emotion_labels = {
    "Angry": 0,
    "Frustration": 1,
    "Boredom": 2,
    "Happy": 3,
    "Sad": 4,
    "Surprise": 5,
    "Neutral": 6,
}
"""
import os
import torch
from custom_dataset import CustomDataset
from resnet import ResEmoteNet

# Check for GPU availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Define globla variables used across this program
BASE_PATH = "data/"
BATCH_SIZE = 16

# ============= OPTIONAL: build a data transformation pipeline ============

# Split dataset into train and test subset
full_ds = CustomDataset(os.path.join(BASE_PATH, "fer2013_filtered.csv"))
train_ds, test_ds = torch.utils.data.random_split(full_ds, [0.7, 0.3])

# Create a dataloader for training set
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_image, train_label = next(iter(train_dl))

# Create a dataloader for test set
test_ds = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
test_image, test_label = next(iter(test_ds))

# Check the dimension of train and test dataset 
print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

# Load a model
model = ResEmoteNet().to(device)

# Show total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {total_params:,}")