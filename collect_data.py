"""
    Collect google images of three additional facial emotions used for training an emotion-aware AI
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
import random

import argparse

def detect_face(image):
    """
        Detects the largest face in an image and returns the cropped face.

        image: PIL image object
    """
    gray = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None  # No face detected
    
    x, y, w, h = faces[0]  # Assume the first detected face is the primary one
    face = image.crop((x, y, x + w, y + h))
    face = face.resize((48, 48), Image.Resampling.LANCZOS) # Apply a smoother downscaling filter
    return face

def get_images_ggl(query, api_key, cse_id, num_images, output_csv, img_class):
    """
        Downloads images from Google Images for a specific query, detects faces,
        processes them into FER2013 format, and appends them to a CSV file.

        query: Google Image search string
        api_key: Google API key
        cse_id: Custom search engine ID
        num_images: Total number of images to collect
        output_csv: name of csv file to store output
        image_class: integer indicating the class lable of emotion
    """
    service = build("customsearch", "v1", developerKey=api_key)
    existing_rows = set()

    if os.path.exists(output_csv):
        with open(output_csv, mode="r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            existing_rows = {row[1] for row in reader}  # Store existing pixel data to avoid duplicates

    with open(output_csv, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(output_csv).st_size == 0:
            writer.writerow(["emotion", "pixels"])  # Write header if file is empty

        queries = [query, query + " face", query + " facial expression", query + " faces men", 
                   query + " faces women", query + " emotion"]
        random.shuffle(queries)  # Randomize queries to get diverse images
        
        total_images_fetched = 0

        for q in queries:
            for start in range(1, 91, 10):  # Google API allows up to start=90 (for 100 results max per query)
                if total_images_fetched >= num_images:
                    return
                try:
                    result = service.cse().list(
                        q=q,
                        cx=cse_id,
                        searchType="image",
                        num=10,
                        start=start,
                        imgSize='medium'
                    ).execute()

                    for item in result.get("items", []):
                        if total_images_fetched >= num_images:
                            return

                        img_url = item["link"]
                        response = requests.get(img_url, timeout=10)
                        img = Image.open(io.BytesIO(response.content))
                        img = ImageOps.grayscale(img)
                        
                        face = detect_face(img)
                        if face is None:
                            continue  # Skip images without detected faces
                        
                        pixel_array = np.array(face).flatten()
                        pixel_str = " ".join(map(str, pixel_array))
                        
                        if pixel_str not in existing_rows:
                            writer.writerow([img_class, pixel_str])
                            existing_rows.add(pixel_str)
                            total_images_fetched += 1
                
                except Exception as e:
                    print(f"Error processing image: {e}")

def verify_images(csv_file, audit=True):
    """
        Opens the saved images from the CSV file and allows the user to accept or reject them.
        If rejected, removes the corresponding row from the CSV file.

        csv_file: csv file where each row consists of an integer label column and image pixel column
        audit: view images without modifying the file if False
    """
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        pixel_values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        image = pixel_values.reshape(48, 48)  # Reshape to 48x48
        
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

        if audit:
            decision = input("Accept this image? (y/n): ").strip().lower()
            if decision == "n":
                df.drop(index, inplace=True)  # Remove the row

    # Save the updated CSV
    df.to_csv(csv_file, index=False)

def parse_arg():
    """
        A utility function that processes command line arguments
    """
    parser = argparse.ArgumentParser(prog="Image Collector", description="Collect, audit or view images")
    parser.add_argument("-mode", nargs="?", default="collect")

    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_arg()
    mode = args.mode

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        raise ValueError("Missing API credentials. Check your .env file.")

    EMOTION_QUERY = "frustration" # Change the query string for other emotions
    NUM_IMAGES = 1000
    OUTPUT_CSV = f"data/fer2013.csv" # Change file name to match search results
    IMG_CLS = 1 # Change to other emotion classes

    if mode == 'collect':
        get_images_ggl(EMOTION_QUERY, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, NUM_IMAGES, OUTPUT_CSV, IMG_CLS)
        print(f"Images processed and saved to {OUTPUT_CSV}")
    elif mode == 'audit':
        verify_images(OUTPUT_CSV)
    elif mode == 'view':
        verify_images(OUTPUT_CSV, False)
