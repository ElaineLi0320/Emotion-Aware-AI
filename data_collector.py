"""
    A class that encapsulates essential functions for collecting images from Google Image 
"""
import os
import csv
import io
import re
import cv2
import numpy as np
import requests
import random
import pandas as pd

from googleapiclient.discovery import build
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class DataCollector:
    # A class attribute to track total number of images collected so far
    img_collected = 0

    def __init__(self, query, api_key, cse_id, csv_file, img_class, num_images=20, base_path="data/"):
        """
            query: a string expressing an emotion the API should search images for
            api_key: Google Image API key(env)
            cse_id: Google custom search engine ID(env)
            csv_file: csv file name storing FER2013-compliant data
            img_class: integer class label for the emotion
            num_images: total number of images to collect
        """
        self.query = query
        self.api_key = api_key
        self.cse_id = cse_id
        self.csv_file = csv_file
        self.img_class = img_class
        self.num_images = num_images
        self.base_path = base_path
        self._csv_fullpath = os.path.join(self.base_path, self.csv_file)

    def collect(self):
        # Compile a batch of 100 images whenever accumulated images 
        if self.img_collected % 100 == 0 and self.img_collected != 0:
            self._concat()

        service = build("customsearch", "v1", developerKey=self.api_key)
        existing_rows = set()
        self._total_fetched = 0

        with open(self._csv_fullpath, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header 
            writer.writerow(["emotion", "pixels"])

            queries = [self.query, self.query + " face", self.query + " facial expression", self.query + " faces men", 
                    self.query + " faces women", self.query + " emotion"]
            random.shuffle(queries)  # Randomize queries to get diverse images

            for q in queries:
                for start in range(1, 91, 10):  # Google API allows up to start=90 (for 100 results max per query)
                    # if self._total_fetched >= self.num_images:
                    #     break
                    try:
                        result = service.cse().list(
                            q=q,
                            cx=self.cse_id,
                            searchType="image",
                            num=10,
                            start=start,
                            imgSize='MEDIUM'
                        ).execute()

                        for item in result.get("items", []):
                            if self._total_fetched >= self.num_images:
                                self.__class__.increment(self._total_fetched)
                                DataCollector.get_img_collected()
                                return

                            img_url = item["link"]
                            response = requests.get(img_url, timeout=10)
                            img = Image.open(io.BytesIO(response.content))
                            img = ImageOps.grayscale(img)
                            
                            face = self._detect_face(img)
                            if face is None:
                                continue  # Skip images without detected faces
                            
                            pixel_array = np.array(face).flatten()
                            pixel_str = " ".join(map(str, pixel_array))
                            
                            if pixel_str not in existing_rows:
                                writer.writerow([self.img_class, pixel_str])
                                existing_rows.add(pixel_str)
                                self._total_fetched += 1
                    
                    except Exception as e:
                        print(f"Error processing image: {e}")

    def _detect_face(self, image):
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

    def _concat(self):
        """
            Combine data in csv files together for duplicate and second quality check 
        """
        # Match file names like "boredom_ggl_1.csv"
        pattern = re.compile(r"\b[a-z]+_[a-z]+_([0-9]+)\.csv\b")

        df_new = pd.DataFrame(columns=["emotion", "pixels"])
        for csv in os.listdir(self.base_path):
            if pattern.fullmatch(csv):
                full_path = os.path.join(self.base_path, csv)
                df = pd.read_csv(full_path)
                df_new = pd.concat([df_new, df], ignore_index=True)
                os.remove(full_path)

        batch_num = self._img_collected // 100
        output_file = f"{self.query}_ggl_bt{batch_num}.csv"
        df_new.to_csv(os.path.join(self.base_path, output_file), index=False)
        print(f"Batch of 100 images has been saved to {output_file}")

    @classmethod
    def increment(cls, amount):
        cls.img_collected += amount

    @classmethod
    def get_img_collected(cls):
        print(f"Running Total: {cls.img_collected}")

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