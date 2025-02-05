import argparse
from dotenv import load_dotenv
import os

from data_collector import DataCollector
from util import verify_images, concat_csv, display_menu, tally

if __name__ == "__main__":
    load_dotenv()

    # Set global variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
    BASE_PATH = "data/"
    EMOTIONS = ["frustration", "boredom"]

    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        raise ValueError("Missing API credentials. Check your .env file.")

    display_menu()
    mode = int(input("Select an option[1-5]: ").strip())
    
    # Branch to different function calls based on user input
    if mode == 1:
        emotion_query = input("\nEmotion: ")
        output_csv = input("Output File: ")
        img_cls = int(input("Image Class: "))
        collector = DataCollector(emotion_query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, 
                              output_csv, img_cls, num_images=50)
        collector.collect()
    elif mode == 2:
        file = input("\nFile to Open: ")
        verify_images(os.path.join(BASE_PATH, file))
    elif mode == 3:
        file = input("\nFile to Open: ")
        verify_images(os.path.join(BASE_PATH, file), False)
    elif mode == 4:
        output_csv = input("\nName of Combined Files: ")
        concat_csv(BASE_PATH, output_csv)
    elif mode == 5:
        tally(EMOTIONS, BASE_PATH)