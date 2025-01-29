from ultralytics import YOLO
import os

version = "2025_01_28"
output_dir = f"/TFG/models/{version}/"
dataset_dir = "/TFG/datasets_labeled/2024_11_28_canicas_dataset/data.yaml"

model = YOLO("yolo11n")
results = model.train(data=dataset_dir, epochs=30, device=0, imgsz=640)
save_path = os.path.join(output_dir, f"{version}_canicas_{base_model}")
model.save(save_path)
