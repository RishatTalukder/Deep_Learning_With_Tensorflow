import os
import random
import shutil

# Define your paths (change these to match your actual folder locations)
SOURCE_DIR = "/home/itvaya/Downloads/food101/images"  # The folder containing all 101 category folders
TARGET_DIR = "/home/itvaya/Desktop/Deep_Learning_With_Tensorflow/4_Convolutional_Neural_network/10_food_classes"       # The new folder you want to create

def copy_random_categories(source, target, num_classes=10):
    # 1. Get all category folder names
    try:
        all_categories = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    except FileNotFoundError:
        print(f"Error: The source directory '{source}' was not found. Please check your path.")
        return

    if len(all_categories) < num_classes:
        print(f"Error: Source only has {len(all_categories)} folders, but you requested {num_classes}.")
        return

    # 2. Randomly select 10 categories
    random.seed(42)
    selected_categories = random.sample(all_categories, num_classes)
    print(f"Successfully selected {num_classes} random classes:")
    for cat in selected_categories:
        print(f" - {cat}")

    # 3. Create the target directory if it doesn't exist
    os.makedirs(target, exist_ok=True)

    # 4. Copy the selected folders and their contents
    print("\nCopying files... This might take a moment.")
    for category in selected_categories:
        source_folder = os.path.join(source, category)
        target_folder = os.path.join(target, category)
        
        # Copy the entire folder structure and files
        shutil.copytree(source_folder, target_folder)
        print(f"✔ Copied: {category}")

    print(f"\nDone! All 10 classes have been copied to '{target}'.")

# Run the script
copy_random_categories(SOURCE_DIR, TARGET_DIR)
