import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

# Suppress warnings & C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = "/content/fish-disease-xai-research"
TEST_DIR = os.path.join(PROJECT_ROOT, "data/processed/test")
CHECKPOINT_PATH = "models/best_EfficientNetV2S.h5"
OUTPUT_DIR = "results/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
NUM_CLASSES = 8

def build_model():
    base_model = EfficientNetV2S(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(CHECKPOINT_PATH)
    return model, base_model

def get_gradcam_heatmap(model, last_conv_layer_name, img_array, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return superimposed_img

def generate_explanations():
    print("Loading model for Grad-CAM generation...")
    model, base_model = build_model()
    
    # Target last convolutional layer in EfficientNetV2S
    last_conv_layer_name = [layer.name for layer in base_model.layers if isinstance(layer, layers.Conv2D)][-1]
    
    class_names = sorted(os.listdir(TEST_DIR))
    print(f"Generating Grad-CAM overlays using layer '{last_conv_layer_name}'...")

    for class_name in class_names:
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        images = os.listdir(class_dir)
        if not images:
            continue
            
        sample_img_path = os.path.join(class_dir, images[0])
        img = load_img(sample_img_path, target_size=IMG_SIZE)
        img_array = img_to
