import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from models import build_model, IMG_SIZE, NUM_CLASSES

CHECKPOINT_PATH = "models/best_ResNet50.h5"
TEST_DIR = "data/processed/test"
OUTPUT_DIR = "results/gradcam_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_and_overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return img_rgb, heatmap_resized, superimposed_rgb

def generate_xai_reports():
    print("Building model structure and loading trained weights...")
    # Re-instantiate model architecture and load weights
    full_model = build_model(backbone_name="ResNet50", num_classes=NUM_CLASSES)
    full_model.load_weights(CHECKPOINT_PATH)
    
    # Locate backbone layer
    resnet_base = full_model.get_layer("resnet50")
    last_conv_layer_name = "conv5_block3_out"
    
    # Build feature extraction model up to last conv layer
    grad_model = tf.keras.models.Model(
        inputs=[resnet_base.inputs],
        outputs=[resnet_base.get_layer(last_conv_layer_name).output]
    )

    class_folders = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    for class_name in class_folders:
        class_path = os.path.join(TEST_DIR, class_name)
        images = glob.glob(os.path.join(class_path, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(class_path, "*.[pP][nN][gG]"))
        
        if not images:
            continue
            
        sample_img_path = images[0]
        
        # Load and preprocess image
        img = tf.keras.utils.load_img(sample_img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())
        
        # Compute gradient-based activation mapping
        with tf.GradientTape() as tape:
            conv_outputs = grad_model(img_preprocessed)
            tape.watch(conv_outputs)
            
            x = full_model.get_layer("global_avg_pool")(conv_outputs)
            x = full_model.get_layer("batch_norm")(x)
            x = full_model.get_layer("dense_features")(x)
            x = full_model.get_layer("dropout")(x)
            preds = full_model.get_layer("predictions")(x)
            
            top_class_idx = tf.argmax(preds[0])
            loss = preds[:, top_class_idx]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        # Generate plot output
        img_orig, heatmap_resized, superimposed = save_and_overlay_gradcam(sample_img_path, heatmap)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_orig)
        axes[0].set_title(f"Original: {class_name[:15]}")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(superimposed)
        axes[2].set_title("Superimposed Focus")
        axes[2].axis("off")
        
        save_path = os.path.join(OUTPUT_DIR, f"GradCAM_{class_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Grad-CAM heatmap for class '{class_name}' -> {save_path}")

    print("\nAll Grad-CAM visualizations generated successfully in 'results/gradcam_visualizations/'!")

if __name__ == "__main__":
    generate_xai_reports()
