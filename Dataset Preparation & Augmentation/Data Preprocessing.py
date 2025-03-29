import cv2
import os
import numpy as np

# Input folder where images are stored
input_folder = "/path/to/Fall/images"  # Change for Non-Fall
output_folder = "/path/to/Preprocessed/Fall/"
os.makedirs(output_folder, exist_ok=True)  # Create output folder

# Loop through each image in the folder
for img_file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_file)
    
    # Read the image
    img = cv2.imread(img_path)

    # Resize to 224x224 pixels
    img_resized = cv2.resize(img, (224, 224))

    # Normalize pixel values (convert range from 0-255 to 0-1)
    img_normalized = img_resized / 255.0

    # Save the preprocessed image (convert back to 0-255 for saving)
    cv2.imwrite(os.path.join(output_folder, img_file), (img_normalized * 255).astype(np.uint8))

print(" Preprocessing Complete! Images are resized and normalized.")
->4th
