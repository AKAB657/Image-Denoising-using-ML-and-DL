import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_generator import SyntheticNoiseDataGenerator  # Updated generator

def build_unet_dncnn(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
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
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# Set dataset directory
dataset_dir = "D:/synthetic_grain"

# Initialize data generators
train_generator = SyntheticNoiseDataGenerator(dataset_dir, batch_size=8, img_size=(256, 256), mode="train")
val_generator = SyntheticNoiseDataGenerator(dataset_dir, batch_size=8, img_size=(256, 256), mode="validate")

# Build model
model = build_unet_dncnn(input_shape=(256, 256, 3))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Save model
model.save('unet_dcnn_denoising_model_rgb_256.keras')
print("Model saved as 'unet_dcnn_denoising_model_rgb_256.keras'")
