import argparse
from dotenv import load_dotenv
import os

from data_collector import DataCollector, verify_images

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
    BASE_PATH = "data/"

    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        raise ValueError("Missing API credentials. Check your .env file.")

    if mode == 'collect':
        emotion_query = input("Emotion: ")
        output_csv = input("Output File: ")
        img_cls = int(input("Image Class: "))
        collector = DataCollector(emotion_query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, 
                              output_csv, img_cls)
        collector.collect()
    elif mode == 'audit':
        file = input("File to Open: ")
        verify_images(os.path.join(BASE_PATH, file))
    elif mode == 'view':
        file = input("File to Open: ")
        verify_images(os.path.join(BASE_PATH, file), False)