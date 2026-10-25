import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = "/content/fish-disease-xai-research"
TEST_DIR = os.path.join(PROJECT_ROOT, "data/processed/test")
CHECKPOINT_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

def build_model(base_model_cls, name):
    base_model = base_model_cls(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs, name=name)
    return model

def evaluate_ensemble():
    print("Building model structures and loading trained weights...")
    
    # Load EfficientNetV2S
    eff_model = build_model(EfficientNetV2S, "EfficientNetV2S")
    eff_model.load_weights(os.path.join(CHECKPOINT_DIR, "best_EfficientNetV2S.h5"))
    
    # Load ResNet50
    resnet_model = build_model(ResNet50, "ResNet50")
    resnet_model.load_weights(os.path.join(CHECKPOINT_DIR, "best_ResNet50.h5"))
    
    # Predict on hold-out Test Set
    print("\nGenerating predictions on Test Set...")
    eff_preds = eff_model.predict(test_generator)
    resnet_preds = resnet_model.predict(test_generator)
    
    # Soft-Voting Ensemble (Average Probabilities)
    ensemble_preds = (eff_preds + resnet_preds) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification Report
    print("\n================ Soft-Voting Ensemble Test Metrics ================")
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)
    
    # Save Report to File
    with open(os.path.join(RESULTS_DIR, "ensemble_test_report.txt"), "w") as f:
        f.write(report)

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Soft-Voting Ensemble (ResNet50 + EfficientNetV2S) Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    evaluate_ensemble()
