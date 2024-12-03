from ultralytics import YOLO
import torch
import os
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

data_path = "/TFG/datasets_labeled/fotos_muy_juntas/data.yaml"
output_dir = "../validation_predictions"

models_paths = [
    "../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_INT8.engine",
                ]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
for model_path in models_paths:
    print(f"VALIDATING MODEL: {model_path}")
    
    model = YOLO(model_path)

    metrics = results = model.val(data=data_path, batch=-1, half=True, plots=True, project=output_dir, conf=0.4, device=0, split='test') 

    print(metrics.box.map)      