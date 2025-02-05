"""
    A class that encapsulates essential functions for collecting images via Google Image API
"""

import os
import csv
import io
import cv2
import numpy as np
import requests
import random

from googleapiclient.discovery import build
from PIL import Image, ImageOps

class DataCollector:

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
        """
            Retrieve images from Google Image via API calls that appends differnt keywords to the basic
            query string 
        """
        service = build("customsearch", "v1", developerKey=self.api_key)
        self._total_fetched = 0
        # Set a plain text file to store urls that are already visited
        url_file = os.path.join(self.base_path, f"{self.query}_img_urls.txt")

        # Read visited urls if the file exists
        if os.path.exists(url_file):
            with open(url_file, "r") as f:
                seen_urls = set(f.read().splitlines())
        else:
            seen_urls = set()

        with open(self._csv_fullpath, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header 
            writer.writerow(["emotion", "pixels"])
            self.create_queries()

            for q in self.queries_:
                # ========== DEBUGGING ============
                print(f"\nQuery selected: {q}")

                start_indices = [random.randint(1, 90) for _ in range(self.num_images // 10)]
                random.shuffle(start_indices)
                # Adjust starting index in each query to reduce duplicates
                for start in start_indices: 
                    # ========== DEBUGGING ===========
                    print(f"Starting index: {start}\n")

                    try:
                        result = service.cse().list(
                            q=q,
                            cx=self.cse_id,
                            searchType="image",
                            num=10,
                            start=start,
                            imgSize='MEDIUM',
                            imgType='face'
                        ).execute()

                        for item in result.get("items", []):
                            if self._total_fetched >= self.num_images:
                                return

                            img_url = item["link"]

                            # Check if the feteched url is already visited
                            if img_url in seen_urls:
                                continue
                            
                            # Otherwise add it to seen_urls
                            seen_urls.add(img_url)
                            with open(url_file, "a") as f:
                                f.write(img_url + "\n")

                            response = requests.get(img_url, timeout=10)
                            img = Image.open(io.BytesIO(response.content))
                            img = ImageOps.grayscale(img)
                            
                            face = self._detect_face(img)
                            if face is None:
                                continue  # Skip images without detected faces
                            
                            pixel_array = np.array(face).flatten()
                            pixel_str = " ".join(map(str, pixel_array))
                            writer.writerow([self.img_class, pixel_str])
                            self._total_fetched += 1
                    
                    except Exception as e:
                        print(f"Error processing image: {e}")

    def create_queries(self):
        """
            Create a diverse query pools to avoid duplicate queries across multiple runs
        """
        self.queries_ = [
                    self.query, self.query + " face", self.query + " facial expression", 
                    self.query + " faces men", self.query + " faces women", 
                    self.query + " emotion", self.query + " student",
                    self.query + " mood", self.query + " headshot", 
                    self.query + " close-up",  self.query + " candid moment", 
                    self.query + " natural emotion", self.query + " portrait"
                ]
        random.shuffle(self.queries_)
        

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

