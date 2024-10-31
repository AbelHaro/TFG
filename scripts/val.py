from ultralytics import YOLO
import torch
import os
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n_FP16.engine"
data_path = "/TFG/datasets_labeled/2024_10_24_canicas_dataset/data.yaml"
output_dir = "../validation_predictions"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = YOLO(model_path)

metrics = results = model.val(data=data_path, batch=-1, save_hybrid=True, save_json=True, half=True, plots=True, project=output_dir, conf=0.2, device=0) 

print(metrics.box.map)      