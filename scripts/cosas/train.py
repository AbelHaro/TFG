from ultralytics import YOLO
import torch
from check_gpu_exists import exists_gpu
import os

# Verificar si hay una GPU disponible
if not exists_gpu():
    print("No GPU found. Exiting...")
    exit()

# Definir el directorio de salida y el conjunto de datos
# version  = "2024_11_28"
version = "2025_02_24"
output_dir = f"/TFG/models/canicas/{version}/"
dataset_dir = f"/TFG/datasets_labeled/{version}_canicas_dataset/data.yaml"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Modelos base a usar para entrenamiento
base_models = [
    "yolov5nu.pt",
    "yolov5mu.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    # "yolo11m.pt",
    # "yolo11l.pt",
    # "yolo11s.pt",
    # "yolov5nu.pt",
]

# Iterar sobre cada modelo base y entrenar
for base_model in base_models:

    print(f"Entrenando el modelo base: {base_model}")

    # Cargar el modelo base
    model = YOLO(base_model)

    # Entrenamiento del modelo
    results = model.train(
        data=dataset_dir,
        epochs=30,
        device=0,
        plots=True,
        name=f"./train_logs/{version}_canicas_{base_model}",
    )

    # Definir la ruta completa para guardar el modelo
    save_path = os.path.join(output_dir, f"{version}_canicas_{base_model}")

    # Guardar el modelo entrenado
    model.save(save_path)

    print(f"Modelo guardado en: {save_path}")


# ultralytics/ultralytics    latest-jetson-jetpack5   78678cfbf7c1   4 weeks ago     13.8GB
# ultralytics/ultralytics    8.3.90-jetson-jetpack5   262224e52266   7 weeks ago     13.8GB
# ultralytics/ultralytics    8.3.72-jetson-jetpack5   dc72d44368f8   3 months ago    13.6GB
# ultralytics/ultralytics    8.3.38-jetson-jetpack5   725d51aac26c   5 months ago    13.8GB

# https://medium.com/@noel.benji/inside-yolo-what-are-c3k2-c2f-c3k-blocks-806ae4cd486f
