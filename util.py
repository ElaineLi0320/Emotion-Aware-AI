"""
    A collection of supporting functions
"""

import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import numpy as np

def verify_images(csv_file, audit=True):
    """
        Opens the saved images from the CSV file and allows the user to accept or reject them.
        If rejected, removes the corresponding row from the CSV file.

        csv_file: csv file where each row consists of an integer label column and image pixel column
        audit: view images without modifying the file if False
    """
    # Make sure provided file exists
    if not os.path.exists(csv_file):
        raise ValueError(f"{csv_file} doesn't exist. Please provide a right file name.")

    df = pd.read_csv(csv_file)
    rows = len(df.index)

    for index, row in df.iterrows():
        pixel_values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        image = pixel_values.reshape(48, 48)  # Reshape to 48x48
        
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(f"Image {index+1}/{rows}")
        plt.show()

        if audit:
            decision = input("Accept this image? (y/n): ").strip().lower()
            if decision == "n":
                df.drop(index, inplace=True)  # Remove the row

    # Save the updated CSV
    if audit:
        df.to_csv(csv_file, index=False)
        print(f"\nSummary: {len(df.index)} out of {rows} were kept.")

def concat_csv(file_dir, output_file):
    """
        Check if there are five/two mini-batch files and combine them for double check 
    """
    # Match file names like "boredom_ggl_1.csv"
    pattern = re.compile(r"\b[a-z]+_[a-z]+_([0-9]+)\.csv\b")
    total_rows = 0

    df_new = pd.DataFrame(columns=["emotion", "pixels"])
    for csv in os.listdir(file_dir):
        if pattern.fullmatch(csv):
            full_path = os.path.join(file_dir, csv)
            df = pd.read_csv(full_path)
            total_rows += len(df.index)
            df_new = pd.concat([df_new, df], ignore_index=True)
            os.remove(full_path)

    full_path = os.path.join(file_dir, output_file)
    if os.path.exists(full_path):
        print("File {full_path} already exists. Operation aborted.")
    else:
        df_new.to_csv(full_path, index=False)
        print(f"A total of {total_rows} images has been saved to {output_file}")

def tally(emo_list, base_path):
    """
        Compute a tally of records for each emotion category

        emo_list: a list of string emotions
    """
    ptn = re.compile(r"\b[a-z]+_[a-z]+_bt([0-9]+)\.csv\b")

    for emo in emo_list:
        running_total = 0
        print(f"\nCategory: {emo}")
        # Get the directory an emotion category is in
        dire = os.path.join(base_path, emo)

        for csv in os.listdir(dire):
            if ptn.fullmatch(csv) and csv.startswith(emo):
                df = pd.read_csv(os.path.join(dire, csv))
                running_total += len(df.index)
        print(f"Total images collected: {running_total}\n")

def show_all_files(dir):
    """
        Display all csv image files in a directory
    """
    print("\nAll files in current directory:")
    for file in os.listdir(dir):
        print(file)

def display_menu():
    """
        Display program menus to users
    """
    print("================= Program Menu ===================")
    print("1) Collect images\n2) Audit images\n3) View images\n4) Combine mini-batch files\n5) Tally images\n6) Cross check")

    
