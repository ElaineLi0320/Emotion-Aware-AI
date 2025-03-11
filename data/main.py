from dotenv import load_dotenv
import os

from Data_Collector import DataCollector
from util import verify_images, concat_csv, display_menu, tally, show_all_files

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
    mode = int(input("Select an option[1-6]: ").strip())
    
    # Branch to different function calls based on user input
    if mode == 1:
        emotion_query = input("\nEmotion: ").strip()
        output_csv = input("Output File: ").strip()
        img_cls = int(input("Image Class: ").strip())
        collector = DataCollector(emotion_query, GOOGLE_API_KEY, CUSTOM_SEARCH_ENGINE_ID, 
                              output_csv, img_cls, num_images=50)
        collector.collect()
    if mode == 2:
        file = input("\nFile to Open: ").strip()
        # [WARNING] Comment out the following line if no splitting of filename is required 
        emo = file.split("_")[0]
        verify_images(os.path.join(BASE_PATH, emo, file))
    elif mode == 3:
        file = input("\nFile to Open: ").strip()
        # [WARNING] Adjsut the following lines if no splitting of filename is required 
        emo = file.split("_")[0]
        verify_images(os.path.join(BASE_PATH, emo, file), False)
    elif mode == 4:
        output_csv = input("\nName of Combined Files: ").strip()
        folder = input("Folder to Search: ").strip()
        concat_csv(os.path.join(BASE_PATH, folder), output_csv)
    elif mode == 5:
        tally(EMOTIONS, BASE_PATH)
    elif mode == 6:
        emo = input("\nEmotion Category: ").strip()
        show_all_files(os.path.join(BASE_PATH, emo))
        file_name = input("File to check: ").strip()
        verify_images(os.path.join(BASE_PATH, emo, file_name))
    else:
        print("Option must be an integer between 1 and 6. Exiting...")
        exit(1)