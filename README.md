# -Indian-Traffic-Sign-Recognition
A robust Two-Stage Traffic Sign Detection &amp; Recognition system specialized for Indian roads. Combines YOLOv8 for real-time object localization and ResNet50 for high-accuracy classification.
# Two-Stage Traffic Sign Detection & Recognition System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)

A hybrid Computer Vision pipeline designed to accurately detect and classify traffic signs in challenging environments (specifically Indian roads). This project solves the "small object classification" problem by decoupling localization from recognition.

## The Architecture

This project uses a **Two-Stage Pipeline** approach:

1.  **Stage 1 (Localization):** A **YOLOv8 Nano** model scans the full scene to find bounding boxes of traffic signs. It focuses purely on finding *where* the sign is, ignoring *what* specific sign it is.
2.  **Processing:** The detected region is cropped with **Context Padding** (10% expansion) to ensure the sign's edges are preserved.
3.  **Stage 2 (Classification):** A **ResNet50** (Transfer Learning) model takes the high-resolution crop and classifies it into specific classes (e.g., "Speed Limit 50", "Stop", "No Entry").

### Why Two Stages?
* **Accuracy:** Single-stage detectors often struggle to distinguish between similar signs (e.g., Speed Limit 30 vs 80) when the object is small.
* **Modularity:** You can retrain the classifier on new sign types without retraining the expensive detector.

Datasets
We utilize two distinct datasets to maximize performance:

Detection Data: Traffic Signs Dataset (Indian Roads)

Used for: Training YOLOv8 to separate "Signs" from "Background".

Features: Full road scenes, varying lighting, Indian road conditions.

Classification Data: Indian Traffic Sign Dataset

Used for: Training ResNet50 to read the sign.

Features: ~15,000 cropped images of 59 specific sign classes.

Installation
Clone the repository:

Bash

git clone [https://github.com/your-username/traffic-sign-two-stage.git](https://github.com/your-username/traffic-sign-two-stage.git)
cd traffic-sign-two-stage
Install Dependencies:

Bash

pip install ultralytics kagglehub tensorflow opencv-python numpy pyyaml
Usage Guide
Follow these steps strictly to build the system from scratch.

Step 1: Setup Detection Data
Download and fix the folder structure for the detection dataset.

Bash

python 01_setup_data.py
Step 2: Train the Detector (Stage 1)
Trains YOLOv8n to find signs. (Approx. time: 15-30 mins on GPU).

Bash

python 02_train_detector.py
Output: Saves model to traffic_project/stage1_detector/weights/best.pt

Step 3: Train the Classifier (Stage 2)
Downloads the classification dataset and fine-tunes ResNet50.

Bash

python 04_train_classifier.py
Output: Saves model to traffic_classifier.keras and generates class_names.txt.

Step 4: Run Inference
Test the full pipeline on a random image from the validation set.

Bash

python 03_inference_pipeline.py
Note: You can edit 03_inference_pipeline.py to point to your own video or specific image path.

Configuration
You can tweak the performance in 03_inference_pipeline.py:

CONFIDENCE_THRESHOLD = 0.4: Lower this if signs are being missed; raise it if you see false positives (trash detected as signs).

Context Padding: In process_image, the code adds 10% padding (pad_x, pad_y) to the bounding box. Increase this if the detector is cutting off the edges of signs.

Results (Baseline)
Stage 1 (YOLOv8n): Fast inference (~10ms on GPU, ~300ms on CPU). High recall for sign regions.

Stage 2 (ResNet50): High accuracy classification on cropped inputs.

## Project Structure

```text
├── 01_setup_data.py           # Automates dataset download & formatting (YOLO fix included)
├── 02_train_detector.py       # Trains YOLOv8 (Stage 1) on Indian Road Scenes
├── 03_inference_pipeline.py   # The "Production" script: runs Detection + Classification
├── 04_train_classifier.py     # Trains ResNet50 (Stage 2) on cropped sign classes
├── traffic_signs.yaml         # YOLO configuration file (Auto-generated)
├── class_names.txt            # List of classes for Stage 2 (Auto-generated)
├── requirements.txt           # Dependencies
└── traffic_project/           # Stores trained models and logs
