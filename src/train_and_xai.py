import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import build_model, IMG_SIZE, NUM_CLASSES

BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

PROCESSED_DATA_DIR = "data/processed"
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_and_explain(backbone_name="ResNet50"):
    print(f"\n--- Starting Training for {backbone_name} ---")
    
    # 1. Data Generators with Augmentation for Training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )

    # 2. Build Model & Callbacks
    model = build_model(backbone_name=backbone_name, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{backbone_name}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    # 3. Train Model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\nTraining for {backbone_name} completed! Best model saved to {checkpoint_path}")
    return history

if __name__ == "__main__":
    train_and_explain("ResNet50")
