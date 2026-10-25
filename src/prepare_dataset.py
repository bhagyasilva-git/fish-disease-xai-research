import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set fixed random seed for reproducibility
SEED = 42
random.seed(SEED)

RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def create_reproducible_split():
    if not RAW_DATA_DIR.exists():
        print(f"Error: Directory '{RAW_DATA_DIR}' does not exist. Please place your raw Kaggle dataset folders in 'data/raw/'.")
        return

    classes = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    if not classes:
        print(f"No subdirectories found in '{RAW_DATA_DIR}'. Please add class folders containing your images.")
        return

    print(f"Found {len(classes)} classes: {classes}")

    for class_name in classes:
        class_dir = RAW_DATA_DIR / class_name
        images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]")) + list(class_dir.glob("*.[jJ][pP][eE][gG]"))
        
        if not images:
            print(f"Warning: No images found in class folder '{class_name}'. Skipping.")
            continue

        # Split into Train vs (Val + Test)
        train_imgs, val_test_imgs = train_test_split(
            images, 
            test_size=(VAL_RATIO + TEST_RATIO), 
            random_state=SEED, 
            shuffle=True
        )
        
        # Split (Val + Test) evenly into Val and Test
        val_imgs, test_imgs = train_test_split(
            val_test_imgs, 
            test_size=0.5, 
            random_state=SEED, 
            shuffle=True
        )

        # Copy images into structured train/val/test split folders
        for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_dir = OUTPUT_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy(img, target_dir / img.name)
                
    print(f"\nSuccess! Dataset reproducibly split into 80% Train, 10% Val, 10% Test using SEED={SEED}.")

if __name__ == "__main__":
    create_reproducible_split()
