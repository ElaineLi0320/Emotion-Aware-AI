"""
    Prepare data for model training
"""
import pandas as pd
import os

emotion_labels = {
    "Angry": 0,
    "Frustration": 1,
    "Boredom": 2,
    "Happy": 3,
    "Sad": 4,
    "Surprise": 5,
    "Neutral": 6,
}
BASE_PATH = "data/"

fer2013 = pd.read_csv(os.path.join(BASE_PATH, "fer2013.csv"))
# Remove disgust and fear images
fer2013_filtered = fer2013[(fer2013.emotion != 1) & (fer2013.emotion != 2)]
# Save filtered df to a new csv file
fer2013_filtered.to_csv(os.path.join(BASE_PATH, "fer2013_filtered.csv"))
