"""
    This script is used to create folders that contains images for each emotion 
    in train, val and test sets from a local csv file. 
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the CSV file into a Pandas DataFrame
csv_path = "fer_neu.csv"
df = pd.read_csv(csv_path)

# Define a dictionary to map emotion codes to labels
emotion_labels = {
    "0": "Angry",
    "1": "Frustration",
    "2": "Boredom",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprise",
    "6": "Neutral",
}

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['emotion'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)

datasets = {"train": train_df, "val": val_df, "test": test_df}

output_folder_path = "data/FER_NEU"

# Create the output folders and subfolders if they do not exist
for usage, data in datasets.items():
    usage_folder_path = os.path.join(output_folder_path, usage)
    os.makedirs(usage_folder_path, exist_ok=True)
    for label in emotion_labels.values():
        subfolder_path = os.path.join(usage_folder_path, label)
        os.makedirs(subfolder_path, exist_ok=True)

# Loop over each dataset and save the images
for usage, data in datasets.items():
    for index, row in data.iterrows():
        pixels = row["pixels"].split()
        img_data = [int(pixel) for pixel in pixels]
        img_array = np.array(img_data).reshape(48, 48)
        img = Image.fromarray(img_array.astype("uint8"), "L")

        # Get the emotion label and determine the output subfolder
        emotion_label = emotion_labels[str(row["emotion"])]
        output_subfolder_path = os.path.join(output_folder_path, usage, emotion_label)
        output_file_path = os.path.join(output_subfolder_path, f"{index}.jpg")
        img.save(output_file_path)

print("Dataset prepared and organized into train, val, and test folders.")
