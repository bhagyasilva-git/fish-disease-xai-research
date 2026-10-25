import os
import shutil
import glob
from sklearn.model_selection import train_test_split

PROJECT_ROOT = "/content/fish-disease-xai-research"
RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
UNIFIED_DIR = os.path.join(PROJECT_ROOT, "data/raw/unified")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
SEED = 42

def prepare_data():
    print(f"Step 1: Merging all pre-split class folders from '{RAW_DIR}' into '{UNIFIED_DIR}'...")

    if os.path.exists(UNIFIED_DIR):
        shutil.rmtree(UNIFIED_DIR)
    os.makedirs(UNIFIED_DIR, exist_ok=True)

    copied_count = 0

    # Recursively traverse raw directory
    for root, dirs, files in os.walk(RAW_DIR):
        if "unified" in root or "processed" in root:
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_file_path = os.path.join(root, file)
                
                # Class name is the immediate parent directory name
                class_name = os.path.basename(root)
                
                target_class_dir = os.path.join(UNIFIED_DIR, class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                
                # Prepend parent folder (e.g. train_ or test_) to avoid file name collisions
                parent_split = os.path.basename(os.path.dirname(root))
                dest_file_name = f"{parent_split}_{file}"
                dest_file_path = os.path.join(target_class_dir, dest_file_name)
                
                shutil.copy(src_file_path, dest_file_path)
                copied_count += 1

    classes = [d for d in os.listdir(UNIFIED_DIR) if os.path.isdir(os.path.join(UNIFIED_DIR, d))]
    
    if not classes or copied_count == 0:
        print(f"\n[Error] No images found inside '{RAW_DIR}'.")
        return

    print(f"\nMerging Complete! Merged {copied_count} total images across {len(classes)} classes:")
    for c in sorted(classes):
        count = len(glob.glob(os.path.join(UNIFIED_DIR, c, "*.*")))
        print(f"  - {c}: {count} total images")

    # Step 2: Stratified 80/10/10 Split with SEED=42
    print("\nStep 2: Performing Stratified Split (80% Train, 10% Val, 10% Test) with SEED=42...")

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    for split in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(os.path.join(PROCESSED_DIR, split, c), exist_ok=True)

    for c in classes:
        class_path = os.path.join(UNIFIED_DIR, c)
        images = glob.glob(os.path.join(class_path, "*.*"))
        
        # 80% Train, 20% Temp (Val + Test)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.20, random_state=SEED)
        
        # 10% Val, 10% Test
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=SEED)
        
        for img_path in train_imgs:
            shutil.copy(img_path, os.path.join(PROCESSED_DIR, "train", c, os.path.basename(img_path)))
        for img_path in val_imgs:
            shutil.copy(img_path, os.path.join(PROCESSED_DIR, "val", c, os.path.basename(img_path)))
        for img_path in test_imgs:
            shutil.copy(img_path, os.path.join(PROCESSED_DIR, "test", c, os.path.basename(img_path)))

    print("\n--- Final Dataset Split Summary ---")
    for split in ["train", "val", "test"]:
        count = sum(len(files) for _, _, files in os.walk(os.path.join(PROCESSED_DIR, split)))
        print(f" -> {split.capitalize()} Set: {count} images")

if __name__ == "__main__":
    prepare_data()
