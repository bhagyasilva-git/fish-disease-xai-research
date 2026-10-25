import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import build_model, IMG_SIZE, NUM_CLASSES

CHECKPOINT_PATH = "models/best_ResNet50.h5"
TEST_DIR = "data/processed/test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_test_set():
    print("Building model structure and loading trained weights...")
    model = build_model(backbone_name="ResNet50", num_classes=NUM_CLASSES)
    model.load_weights(CHECKPOINT_PATH)

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    print("\nEvaluating model on held-out test dataset...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 1. Generate Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print("\n--- Quantitative Evaluation Report ---")
    print(report)

    # Save classification report text file
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix - ResNet50 Baseline")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix plot to '{cm_path}'")

if __name__ == "__main__":
    evaluate_test_set()
