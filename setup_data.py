import kagglehub
import os
import shutil
import yaml
import random
from pathlib import Path

def setup_dataset():
    print("[INFO] [1/3] Downloading Dataset via KaggleHub...")
    try:
        # Download the dataset
        path = kagglehub.dataset_download("kaustubhrastogi17/traffic-signs-dataset-indian-roads")
        dataset_root = Path(path)
        print(f"[SUCCESS] Dataset downloaded to: {dataset_root}")
    except Exception as e:
        print(f"[ERROR] Error downloading: {e}")
        return

    # --- FIX: Inspect and Reorganize Structure ---
    print("[INFO] Inspecting dataset structure...")
    
    # Check if 'images' folder exists
    images_source = dataset_root / "images"
    labels_source = dataset_root / "labels"
    
    if not images_source.exists():
        # Sometimes datasets are nested. Search for 'images'
        found = list(dataset_root.rglob("images"))
        if found:
            images_source = found[0]
            labels_source = images_source.parent / "labels"
            print(f"   Found images at: {images_source}")
        else:
            print("[ERROR] Critical Error: Could not find 'images' folder in dataset.")
            return

    # Check if 'train' subfolder already exists inside images
    if (images_source / "train").exists():
        print("   [INFO] Dataset already has train/val split. Using existing structure.")
        train_img_dir = images_source / "train"
        val_img_dir = images_source / "val" 
        
        if not val_img_dir.exists() and (images_source / "valid").exists():
            val_img_dir = images_source / "valid"
            
    else:
        print("   [WARN] Dataset is flat (no train/val split). Reorganizing now...")
        
        # 1. Get all image files
        image_files = [f for f in os.listdir(images_source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)
        
        # 2. Split 80/20
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"   Moving {len(train_files)} to train and {len(val_files)} to val...")

        # 3. Create Directories
        (images_source / "train").mkdir(parents=True, exist_ok=True)
        (images_source / "val").mkdir(parents=True, exist_ok=True)
        (labels_source / "train").mkdir(parents=True, exist_ok=True)
        (labels_source / "val").mkdir(parents=True, exist_ok=True)

        # 4. Move Files
        for f in train_files:
            try:
                # Move Image
                shutil.move(str(images_source / f), str(images_source / "train" / f))
                # Move Label
                label_name = os.path.splitext(f)[0] + ".txt"
                if (labels_source / label_name).exists():
                    shutil.move(str(labels_source / label_name), str(labels_source / "train" / label_name))
            except Exception as e:
                # pass if file already moved or permission error
                pass

        for f in val_files:
            try:
                # Move Image
                shutil.move(str(images_source / f), str(images_source / "val" / f))
                # Move Label
                label_name = os.path.splitext(f)[0] + ".txt"
                if (labels_source / label_name).exists():
                    shutil.move(str(labels_source / label_name), str(labels_source / "val" / label_name))
            except Exception as e:
                pass
        
        print("   [SUCCESS] Reorganization complete.")
        train_img_dir = images_source / "train"
        val_img_dir = images_source / "val"

    # --- Generate YAML ---
    print("[INFO] Generating traffic_signs.yaml...")
    
    data_yaml = {
        'path': str(dataset_root.absolute()), 
        'train': str(train_img_dir.relative_to(dataset_root)),
        'val': str(val_img_dir.relative_to(dataset_root)),
        'nc': 5,
        'names': ['Regulatory', 'Mandatory', 'Informatory', 'General', 'Warning']
    }

    yaml_path = 'traffic_signs.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"[SUCCESS] Configuration saved to '{yaml_path}'")
    print("[DONE] Setup complete. Now run '02_train_detector.py'.")

if __name__ == "__main__":
    setup_dataset()