import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def process_and_save_images(input_folder, output_folder, model_path, img_size=(256, 256), blend_ratio=0.7):
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        # Load the image and preprocess
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read {input_path}. Skipping...")
            continue

        # Convert to RGB for consistency with the model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, img_size) / 255.0
        input_data = np.expand_dims(img_resized, axis=0)

        # Predict using the model
        denoised = model.predict(input_data)[0]

        # Post-process output
        denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        denoised_resized = cv2.resize(denoised, (img.shape[1], img.shape[0]))

        # Convert back to BGR for saving
        denoised_bgr = cv2.cvtColor(denoised_resized, cv2.COLOR_RGB2BGR)

        # Apply edge-preserving filter for sharpness enhancement
        enhanced = cv2.detailEnhance(denoised_bgr, sigma_s=10, sigma_r=0.15)

        # Blend the enhanced image with the original for detail retention
        blended = cv2.addWeighted(denoised_bgr, blend_ratio, img, 1 - blend_ratio, 0)

        # Save the denoised image with "_denoised" appended to the filename
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_denoised{ext}")
        cv2.imwrite(output_path, blended)

        print(f"Processed and saved: {output_path}")

# Updated paths
input_folder = r"D:\test_img\noisy images"
output_folder = r"D:\test_img\denoised_outputs"
model_path = r"D:\syn3\unet_dcnn_denoising_model_rgb_256.keras"

process_and_save_images(input_folder, output_folder, model_path)
