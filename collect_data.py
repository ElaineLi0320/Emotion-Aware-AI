"""
    Collect images of three additional facial emotions used for training an emotion-aware AI
"""

import os
import csv
import io
from googleapiclient.discovery import build
from PIL import Image, ImageOps
import numpy as np

def download_and_process_images(query, api_key, cse_id, num_images, output_csv):
    """
    Downloads images from Google Images for a specific query, processes them into FER2013 format,
    and saves them in a CSV file.

    Args:
        query (str): Search query (e.g., "boredom face").
        api_key (str): Google API key.
        cse_id (str): Google Custom Search Engine ID.
        num_images (int): Number of images to retrieve.
        output_csv (str): Path to the output CSV file.
    """
    # Initialize Google Custom Search
    service = build("customsearch", "v1", developerKey=api_key)

    # Prepare CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["emotion", "pixels"])  # Write the header

        # Loop to fetch images
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
                    # Download image
                    img_url = item["link"]
                    img = Image.open(io.BytesIO(requests.get(img_url, timeout=10).content))

                    # Convert to grayscale, resize to 48x48
                    img = ImageOps.grayscale(img)
                    img = img.resize((48, 48))

                    # Convert to pixel array and flatten
                    pixel_array = np.array(img).flatten()
                    pixel_str = " ".join(map(str, pixel_array))

                    # Write to CSV with label (assuming "boredom" is label 7 for this example)
                    writer.writerow([7, pixel_str])

                except Exception as e:
                    print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Replace with your Google API key and Custom Search Engine ID
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    CUSTOM_SEARCH_ENGINE_ID = "YOUR_CSE_ID"

    # Define search parameters
    EMOTION_QUERY = "boredom face"
    NUM_IMAGES = 50
    OUTPUT_CSV = "boredom_images.csv"

    # Run the function
    download_and_process_images(EMOTION_QUERY, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, NUM_IMAGES, OUTPUT_CSV)

    print(f"Images processed and saved to {OUTPUT_CSV}")