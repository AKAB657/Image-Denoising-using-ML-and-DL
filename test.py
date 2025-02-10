# Test/Inference Script
def process_and_save_images(input_folder, output_folder, model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path, custom_objects={"custom_loss": custom_loss, "psnr_metric": psnr_metric, "ssim_metric": ssim_metric})

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load the image and preprocess
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read {input_path}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256)) / 255.0
        input_data = np.expand_dims(img_resized, axis=0)

        # Predict using the model
        denoised = model.predict(input_data)[0]

        # Post-process output
        denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        denoised_resized = cv2.resize(denoised, (img.shape[1], img.shape[0]))
        denoised_clahe = apply_clahe(denoised_resized)

        # Convert back to RGB for saving
        denoised_bgr = cv2.cvtColor(denoised_clahe, cv2.COLOR_RGB2BGR)

        # Save the denoised image with "_denoised" appended to the filename
        output_path = os.path.join(output_folder, filename.replace('.', '_denoised.'))
        cv2.imwrite(output_path, denoised_bgr)

        print(f"Processed and saved: {output_path}")

# Test Configuration
input_folder = "D:/test_img/noisy images"
output_folder = "D:/test_img/denoised_outputs"
model_path = "D:/synthetic_grain/unet_dcnn_denoising_model_improved.keras"
process_and_save_images(input_folder, output_folder, model_path)
