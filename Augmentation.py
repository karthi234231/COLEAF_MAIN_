import os
import zipfile
import gdown
import cv2
import shutil 
import numpy as np
import albumentations as A
from albumentations.core.composition import OneOf

# Folder to store the extracted images
extracted_folder = "extracted_images"
augmented_folder = "augmented_images"

# Ensure output folders exist
os.makedirs(extracted_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)

# List of zip file links
zip_file_urls = [
    "https://drive.google.com/uc?id=1XaICUD-B5YexqiQ1QKlYYs0kjPIoZCF-",
    "https://drive.google.com/uc?id=10JEKLxxmLkiuX6K_zYljaZd3QrjWxuCJ",
    "https://drive.google.com/uc?id=12Oy6Q3VrU6r03SdFIn-G74dMW8iKucXJ",
    "https://drive.google.com/uc?id=1KbdOg5AFCPAJYcXvAEKQ0XNqZdpgwA3U",
    "https://drive.google.com/uc?id=1SAp-3XbSrRT-g6YR8L9P_cK-0tZ2YgO_",
    "https://drive.google.com/uc?id=11_-ZdS9L0LsiU1GSsxcYQyFbFwZed3Nm",
    "https://drive.google.com/uc?id=17IO7mBskF9QtxIHHepX2cMDzsKRKt2Go",
    "https://drive.google.com/uc?id=1x3QLUx97eeLx8PmsKroX0NjP7yUuCNlM",
    "https://drive.google.com/uc?id=1Wo80zPK-ims0LDuQCIqPkQFQewSlY-IN",
    "https://drive.google.com/uc?id=1jg1DvwEZAAiPg6gjtjAL-u8Npu4hMTs2"
]
# Function to download and extract zip file
def download_and_extract_zip(url, destination_folder):
    zip_file_path = os.path.join(destination_folder, url.split('=')[-1] + ".zip")
    
    if not os.path.exists(zip_file_path):
        print(f"⬇ Downloading {zip_file_path}...")
        gdown.download(url, zip_file_path, quiet=False)

    if not os.path.exists(zip_file_path):
        print(f"❌ Failed to download {zip_file_path}.")
        return

    print(f"📦 Extracting {zip_file_path}...")
    if zipfile.is_zipfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"✅ Extracted {zip_file_path}")
    else:
        print(f"❌ {zip_file_path} is not a valid ZIP file!")

# Download and extract all zip files
for url in zip_file_urls:
    download_and_extract_zip(url, extracted_folder)

# Load all image paths and maintain subfolder mapping
image_files = {}
for root, _, files in os.walk(extracted_folder):  # Recursively search for images
    for f in files:
        if f.endswith(('.jpg', '.png', '.jpeg')):
            folder_path = os.path.relpath(root, extracted_folder)  # Get subfolder path
            if folder_path not in image_files:
                image_files[folder_path] = []
            image_files[folder_path].append(os.path.join(root, f))

original_count = sum(len(files) for files in image_files.values())

if original_count == 0:
    print("⚠️ No images found after extraction!")
    exit()

print(f"✅ Found {original_count} images across {len(image_files)} categories.")

# **Copy Original Images to Augmented Folder**
for folder_path, images in image_files.items():
    aug_folder = os.path.join(augmented_folder, f"aug_{folder_path}")  # Create augmented subfolder
    os.makedirs(aug_folder, exist_ok=True)  # Ensure the folder exists

    for img_path in images:
        shutil.copy(img_path, aug_folder)  # Copy original images

print("✅ Original images copied to augmented folders.")

# Define Augmentation Pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.GaussianBlur(p=0.4),
        A.GaussNoise(p=0.3),
    ], p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(p=0.4),
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), rotate=(-30, 30), p=0.7),  
])

# Adjust dataset size to 1500 images
target_count = 1500
current_total = original_count
needed_images = target_count - current_total

if needed_images <= 0:
    print("✅ Dataset already has 1500 or more images. No augmentation needed.")
    exit()

index = 1
new_count = 0

# Step 1: **Generate at least one augmentation per image in each category**
for folder_path, images in image_files.items():
    aug_folder = os.path.join(augmented_folder, f"aug_{folder_path}")  # Augmented subfolder
    os.makedirs(aug_folder, exist_ok=True)  # Ensure subfolder exists

    print(f"🎯 Generating augmented images for '{folder_path}'...")
    
    for img_path in images:
        if new_count >= needed_images:
            break

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Skipping {img_path} (Could not load image)")
            continue

        # Convert BGR to RGB for augmentation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = augmentations(image=image)["image"]

        # Convert back to BGR for saving
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        # Save new image
        new_filename = f"aug_{index}.jpg"
        new_img_path = os.path.join(aug_folder, new_filename)

        cv2.imwrite(new_img_path, augmented)

        new_count += 1
        index += 1

# Step 2: **Distribute remaining images if total count is still <1500**
remaining_images = target_count - (current_total + new_count)

if remaining_images > 0:
    print(f"🔄 Distributing extra {remaining_images} images evenly...")

    category_list = list(image_files.keys())
    i = 0  # Category index

    while remaining_images > 0:
        folder_path = category_list[i % len(category_list)]  # Rotate through categories
        images = image_files[folder_path]
        aug_folder = os.path.join(augmented_folder, f"aug_{folder_path}")

        # Pick a random image for additional augmentation
        img_path = np.random.choice(images)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augmentations(image=image)["image"]
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        new_filename = f"aug_{index}.jpg"
        new_img_path = os.path.join(aug_folder, new_filename)

        cv2.imwrite(new_img_path, augmented)

        new_count += 1
        remaining_images -= 1
        index += 1
        i += 1  # Move to the next category

print(f"✅ Dataset successfully increased to 1500 images! Augmented images are stored in '{augmented_folder}'.")