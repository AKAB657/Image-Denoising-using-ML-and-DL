import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class SyntheticNoiseDataGenerator(Sequence):
    def __init__(self, base_dir, batch_size=8, img_size=(256, 256), mode="train"):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode
        self.noisy_images = []
        self.clean_images = []
        self._load_image_pairs()

    def _load_image_pairs(self):
        noisy_dir = os.path.join(self.base_dir, self.mode, 'noisy images')
        clean_dir = os.path.join(self.base_dir, self.mode, 'ground truth')
        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))

        for noisy_img in noisy_images:
            base_name = '_'.join(noisy_img.split('_')[1:])  # Remove the noise prefix
            clean_img = base_name
            noisy_path = os.path.join(noisy_dir, noisy_img)
            clean_path = os.path.join(clean_dir, clean_img)

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                self.noisy_images.append(noisy_path)
                self.clean_images.append(clean_path)

    def __len__(self):
        # Use ceil to include the last smaller batch
        return int(np.ceil(len(self.noisy_images) / self.batch_size))

    def __getitem__(self, index):
        batch_noisy_images = self.noisy_images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_clean_images = self.clean_images[index * self.batch_size:(index + 1) * self.batch_size]

        noisy_batch = np.zeros((len(batch_noisy_images), *self.img_size, 3), dtype=np.float32)
        clean_batch = np.zeros((len(batch_clean_images), *self.img_size, 3), dtype=np.float32)

        for i, (noisy_path, clean_path) in enumerate(zip(batch_noisy_images, batch_clean_images)):
            noisy_img = cv2.imread(noisy_path)
            clean_img = cv2.imread(clean_path)
            if noisy_img is None or clean_img is None:
                print(f"Skipping corrupted files: {noisy_path}, {clean_path}")
                continue
            # Convert to RGB and resize
            noisy_batch[i] = cv2.resize(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB), self.img_size) / 255.0
            clean_batch[i] = cv2.resize(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB), self.img_size) / 255.0

        return noisy_batch, clean_batch
