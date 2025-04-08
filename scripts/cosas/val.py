from ultralytics import YOLO
import torch
import os
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

VERSION = "2025_02_24"

data_path = f"/TFG/datasets_labeled/{VERSION}_canicas_dataset/data.yaml"
output_dir = "../validation_predictions"

models_paths = [
    f"../../models/canicas/{VERSION}/{VERSION}_canicas_yolo11n.pt",
    f"../../models/canicas/{VERSION}/{VERSION}_canicas_yolo11n_INT8_GPU.engine",
    f"../../models/canicas/{VERSION}/{VERSION}_canicas_yolo11n_INT8_DLA0.engine",
]


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for model_path in models_paths:
    print(
        "====================================================================================================="
    )
    print(f"VALIDATING MODEL: {model_path}")

    model = YOLO(model_path)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    name = f"val_{model_name}"
    
    metrics = model.val(
        data=data_path,
        batch=1,
        half=True,
        plots=False,
        project=output_dir,
        name=name,
        conf=0.25,
        iou=0.5,
        device=0,
        verbose=True
    )

    # Imprimir métricas detalladas
    print("\nMétricas detalladas:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precisión: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
