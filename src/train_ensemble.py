import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Configs
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 8
PROCESSED_DIR = "/content/fish-disease-xai-research/data/processed"
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(PROCESSED_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(PROCESSED_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

def build_model(base_model_cls, name):
    base_model = base_model_cls(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = True
    
    # Freeze initial layers for stable fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model_cls, model_name):
    print(f"\n--- Training {model_name} ---")
    model = build_model(model_cls, model_name)
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.h5")
    
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return model

def evaluate_ensemble():
    print("\n--- Step 3: Evaluating Soft-Voting Ensemble (ResNet50 + EfficientNetV2S) ---")
    
    eff_model = build_model(EfficientNetV2S, "EfficientNetV2S")
    eff_model.load_weights(os.path.join(CHECKPOINT_DIR, "best_EfficientNetV2S.h5"))
    
    resnet_model = build_model(ResNet50, "ResNet50")
    resnet_model.load_weights(os.path.join(CHECKPOINT_DIR, "best_ResNet50.h5"))
    
    eff_preds = eff_model.predict(val_generator)
    resnet_preds = resnet_model.predict(val_generator)
    
    # Soft-Voting Average
    ensemble_preds = (eff_preds + resnet_preds) / 2.0
    ensemble_acc = np.mean(np.argmax(ensemble_preds, axis=1) == val_generator.classes)
    
    print(f"\n Validation Set Results:")
    print(f" -> EfficientNetV2S Accuracy: {np.mean(np.argmax(eff_preds, axis=1) == val_generator.classes):.4f}")
    print(f" -> ResNet50 Accuracy:        {np.mean(np.argmax(resnet_preds, axis=1) == val_generator.classes):.4f}")
    print(f" -> Soft-Voting Ensemble Acc: {ensemble_acc:.4f}")

if __name__ == "__main__":
    train_model(EfficientNetV2S, "EfficientNetV2S")
    train_model(ResNet50, "ResNet50")
    evaluate_ensemble()
