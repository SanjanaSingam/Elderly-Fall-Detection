import albumentations as A
import cv2
import os
import numpy as np
from albumentations.pytorch import ToTensorV2

# Define transformations (Flip, Rotate, Brightness change)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Flip the image left-right
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness randomly
    A.Rotate(limit=20, p=0.5),  # Rotate image up to ±20 degrees
    ToTensorV2()  # Convert image to tensor for training
])

# Folder paths
input_folder = "/path/to/Preprocessed/Fall/"
output_folder = "/path/to/Augmented/Fall/"
os.makedirs(output_folder, exist_ok=True)

# Loop through images and apply augmentation
for img_file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)

    augmented = transform(image=img)["image"]  # Apply transformations

    # Convert tensor to NumPy for saving
    img_augmented = augmented.numpy().transpose(1, 2, 0)  # Rearrange dimensions
    
    cv2.imwrite(os.path.join(output_folder, "aug_" + img_file), img_augmented)

print(" Data Augmentation Complete! More images generated.")
