import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from pathlib import Path
import os
import numpy as np

def train_classifier():
    print("[INFO] [1/3] Downloading Classification Dataset...")
    try:
        # Download the specific classification dataset (Cropped images)
        path = kagglehub.dataset_download("neelpratiksha/indian-traffic-sign-dataset")
        dataset_root = Path(path)
        print(f"[SUCCESS] Dataset downloaded to: {dataset_root}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    # Locate the actual 'Images' folder inside the download
    # The dataset usually has a structure like: root/Images/0, root/Images/1...
    data_dir = None
    for root, dirs, files in os.walk(dataset_root):
        if "Images" in dirs:
            data_dir = Path(root) / "Images"
            break
            
    # Fallback search if "Images" folder isn't explicitly named
    if not data_dir:
        # Look for any folder that has numbered subfolders (0, 1, 2...)
        for root, dirs, files in os.walk(dataset_root):
            if any(d.isdigit() for d in dirs):
                data_dir = Path(root)
                break

    if not data_dir:
        print("[ERROR] Could not find class folders in the dataset.")
        return

    print(f"[INFO] Training data located at: {data_dir}")

    # --- SETUP PARAMETERS ---
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224) # Standard for ResNet
    EPOCHS = 5           # Keep it low for testing. Increase to 20 for real accuracy.

    # --- LOAD DATA ---
    print("[INFO] Loading images...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Found {num_classes} classes: {class_names}")

    # Save class names to a file so inference can use them later
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # --- OPTIMIZATION ---
    # buffer prefetching for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- BUILD MODEL (ResNet50) ---
    print("[INFO] Building ResNet50 Model...")
    
    # 1. Load Pre-trained Base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze base for first pass

    # 2. Add Custom Head
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)), # Normalize 0-255 to 0-1
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # --- TRAIN ---
    print("[INFO] Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # --- SAVE ---
    save_path = 'traffic_classifier.keras' # .keras is the new standard format
    model.save(save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}")
    print("[INFO] Class names saved to: class_names.txt")

if __name__ == "__main__":
    train_classifier()