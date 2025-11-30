from ultralytics import YOLO
import os

def train_stage1():
    print("[INFO] [2/3] Starting Stage 1 Training (Object Detection)...")
    
    # Check if yaml exists
    if not os.path.exists('traffic_signs.yaml'):
        print("[ERROR] 'traffic_signs.yaml' not found. Please run 01_setup_data.py first.")
        return

    # 1. Load the Model
    model = YOLO('yolov8n.pt') 

    # 2. Train the Model
    results = model.train(
        data='traffic_signs.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        project='traffic_project',
        name='stage1_detector',
        exist_ok=True,
        patience=10
    )
    
    # 3. Export Information
    best_weight_path = os.path.join('traffic_project', 'stage1_detector', 'weights', 'best.pt')
    print(f"\n[SUCCESS] Training Complete!")
    print(f"[INFO] Best model saved at: {best_weight_path}")
    print("[INFO] Use this path in the inference script.")

if __name__ == "__main__":
    train_stage1()