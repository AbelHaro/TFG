from ultralytics import YOLO
import torch
import os
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

data_path = "/TFG/datasets_labeled/fotos_muy_juntas/val.yaml"
output_dir = "../validation_predictions"

models_paths = [
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11s_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11m_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11l_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11x_FP16.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_INT8.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11s_INT8.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11m_INT8.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11l_INT8.engine",
    "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11x_INT8.engine",
                ]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
for model_path in models_paths:
    print("=====================================================================================================")
    print(f"VALIDATING MODEL: {model_path}")
    
    model = YOLO(model_path)

    metrics = results = model.val(data=data_path, batch=16, half=True, plots=True, project=output_dir, conf=0.4, device=0, split='test') 

    print(metrics.box.map)      