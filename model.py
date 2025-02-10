import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import log
from tensorflow.keras.callbacks import EarlyStopping
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications import VGG16
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import albumentations as A

# Enable mixed precision training for better performance
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Data Generator with Advanced Augmentation and Dynamic Image Support
class SyntheticNoiseDataGenerator(Sequence):
    def __init__(self, base_dir, batch_size=8, img_size=(None, None), mode="train"):
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
            base_name = '_'.join(noisy_img.split('_')[1:])  # Remove noise prefix
            clean_img = base_name
            noisy_path = os.path.join(noisy_dir, noisy_img)
            clean_path = os.path.join(clean_dir, clean_img)

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                self.noisy_images.append(noisy_path)
                self.clean_images.append(clean_path)

    def __len__(self):
        return int(np.floor(len(self.noisy_images) / self.batch_size))

    def augment_image(self, image):
        transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1))
        ])
        augmented = transform(image=image)
        return augmented["image"]

    def __getitem__(self, index):
        batch_noisy_images = self.noisy_images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_clean_images = self.clean_images[index * self.batch_size:(index + 1) * self.batch_size]

        noisy_batch = []
        clean_batch = []

        for noisy_path, clean_path in zip(batch_noisy_images, batch_clean_images):
            noisy_img = cv2.imread(noisy_path)
            clean_img = cv2.imread(clean_path)

            if noisy_img is None or clean_img is None:
                continue

            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB) / 255.0
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) / 255.0

            noisy_img = self.augment_image((noisy_img * 255).astype(np.uint8)) / 255.0
            clean_img = self.augment_image((clean_img * 255).astype(np.uint8)) / 255.0

            noisy_batch.append(noisy_img)
            clean_batch.append(clean_img)

        return np.array(noisy_batch), np.array(clean_batch)

# Custom Loss Function with Perceptual and Total Variation Loss
def total_variation_loss(y_pred):
    return tf.reduce_sum(tf.image.total_variation(y_pred))

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    vgg.trainable = False
    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.square(true_features - pred_features))

def custom_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    tv_loss = total_variation_loss(y_pred)
    perceptual = perceptual_loss(y_true, y_pred)
    return mse_loss + 0.1 * tv_loss + 0.01 * perceptual

# Custom Metrics

def psnr_metric(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * log(max_pixel**2 / tf.reduce_mean(tf.square(y_pred - y_true))) / log(10.0)

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Model Definition with Attention Mechanisms and Variable Image Dimensions
def squeeze_excite_block(input_tensor):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // 16, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.multiply([input_tensor, se])

def build_unet_dncnn(input_shape=(None, None, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = squeeze_excite_block(conv3)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up1 = layers.UpSampling2D((2, 2))(conv4)
    up1 = layers.concatenate([up1, conv3])
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up2 = layers.UpSampling2D((2, 2))(conv5)
    up2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up3 = layers.UpSampling2D((2, 2))(conv6)
    up3 = layers.concatenate([up3, conv1])
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv7)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)), loss=custom_loss, metrics=['mse', psnr_metric, ssim_metric])
    return model

# Post-Processing Improvements
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply((l * 255).astype(np.uint8)) / 255.0
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Training Configuration
dataset_dir = "D:/synthetic_grain"
train_generator = SyntheticNoiseDataGenerator(dataset_dir, batch_size=8, mode="train")
val_generator = SyntheticNoiseDataGenerator(dataset_dir, batch_size=8, mode="validate")

model = build_unet_dncnn()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(train_generator, validation_data=val_generator, epochs=50, callbacks=[early_stopping])

# Save the Model
model.save('unet_dcnn_denoising_model_improved.keras')
print("Model saved as 'unet_dcnn_denoising_model_improved.keras'")