import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
from models import build_model, IMG_SIZE, NUM_CLASSES

PROCESSED_DATA_DIR = "data/processed"
CHECKPOINT_DIR = "models"
RESULTS_DIR = "results"
SEED = 42

def train_and_eval_efficientnet():
    print("\n--- Step 1: Training EfficientNetV2S ---")
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        seed=SEED
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        seed=SEED
    )

    model = build_model(backbone_name="EfficientNetV2S", num_classes=NUM_CLASSES, trainable=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_EfficientNetV2S.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)
    print("EfficientNetV2S training complete!")

def evaluate_ensemble():
    print("\n--- Step 2: Evaluating Soft-Voting Ensemble (ResNet50 + EfficientNetV2S) ---")
    
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )
    
    # Load ResNet50
    model_resnet = build_model(backbone_name="ResNet50", num_classes=NUM_CLASSES)
    model_resnet.load_weights(os.path.join(CHECKPOINT_DIR, "best_ResNet50.h5"))
    preds_resnet = model_resnet.predict(test_gen)
    
    # Load EfficientNetV2S
    model_effnet = build_model(backbone_name="EfficientNetV2S", num_classes=NUM_CLASSES)
    model_effnet.load_weights(os.path.join(CHECKPOINT_DIR, "best_EfficientNetV2S.h5"))
    preds_effnet = model_effnet.predict(test_gen)
    
    # Soft-Voting Probability Averaging
    ensemble_preds = (preds_resnet + preds_effnet) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    
    print(f"\n==========================================")
    print(f" Ensemble Model Test Accuracy: {acc*100:.2f}%")
    print(f"==========================================\n")
    print(report)

    with open(os.path.join(RESULTS_DIR, "ensemble_classification_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    train_and_eval_efficientnet()
    evaluate_ensemble()
