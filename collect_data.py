"""
    Collect images of three additional facial emotions used for training an emotion-aware AI
"""

import os
import csv
import io
import cv2
import numpy as np
import requests
from googleapiclient.discovery import build
from PIL import Image, ImageOps
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_face(image):
    """Detects the largest face in an image and returns the cropped face."""
    gray = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None  # No face detected
    
    x, y, w, h = faces[0]  # Assume the first detected face is the primary one
    face = image.crop((x, y, x + w, y + h))
    face = face.resize((48, 48))
    return face

def download_and_process_images(query, api_key, cse_id, num_images, output_csv):
    """
    Downloads images from Google Images for a specific query, detects faces,
    processes them into FER2013 format, and saves them in a CSV file.
    """
    service = build("customsearch", "v1", developerKey=api_key)

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["emotion", "pixels"])  # Write the header

        for start in range(1, num_images + 1, 10):
            result = service.cse().list(
                q=query,
                cx=cse_id,
                searchType="image",
                num=min(10, num_images - start + 1),
                start=start
            ).execute()

            for item in result.get("items", []):
                try:
                    img_url = item["link"]
                    response = requests.get(img_url, timeout=10)
                    img = Image.open(io.BytesIO(response.content))
                    img = ImageOps.grayscale(img)
                    
                    face = detect_face(img)
                    if face is None:
                        continue  # Skip images without detected faces
                    
                    pixel_array = np.array(face).flatten()
                    pixel_str = " ".join(map(str, pixel_array))
                    writer.writerow([7, pixel_str])
                
                except Exception as e:
                    print(f"Error processing image: {e}")

def verify_images(csv_file):
    """
    Opens the saved images from the CSV file and allows the user to accept or reject them.
    If rejected, removes the corresponding row from the CSV file.
    """
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        pixel_values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        image = pixel_values.reshape(48, 48)  # Reshape to 48x48
        
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

        decision = input("Accept this image? (y/n): ").strip().lower()
        if decision == "n":
            df.drop(index, inplace=True)  # Remove the row

    # Save the updated CSV
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        raise ValueError("Missing API credentials. Check your .env file.")

    EMOTION_QUERY = "boredom face"
    NUM_IMAGES = 5
    OUTPUT_CSV = "boredom_images.csv"

    # download_and_process_images(EMOTION_QUERY, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, NUM_IMAGES, OUTPUT_CSV)

    # print(f"Images processed and saved to {OUTPUT_CSV}")

    verify_images(OUTPUT_CSV)
