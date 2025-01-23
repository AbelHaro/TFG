from ultralytics import YOLO
import torch
from check_gpu_exists import exists_gpu
import os

# Verificar si hay una GPU disponible
if not exists_gpu():
    print("No GPU found. Exiting...")
    exit()

# Definir el directorio de salida y el conjunto de datos
version = "2024_11_28"
output_dir = f"/TFG/models/canicas/{version}/"
dataset_dir = "/TFG/datasets_labeled/2024_11_28_canicas_dataset/data.yaml"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Modelos base a usar para entrenamiento
base_models = [
    "yolo11n.pt",
    #"yolo11s.pt",
    #"yolo11m.pt",
    #"yolo11l.pt",
    #"yolo11x.pt",
    #"yolov5nu.pt",
]

# Iterar sobre cada modelo base y entrenar
for base_model in base_models:
    
    print(f"Entrenando el modelo base: {base_model}")
    
    # Cargar el modelo base
    model = YOLO(base_model)

    # Entrenamiento del modelo
    results = model.train(data=dataset_dir, epochs=30, device=0, imgsz=640)
    
    # Definir la ruta completa para guardar el modelo
    save_path = os.path.join(output_dir, f"{version}_canicas_{base_model}")

    # Guardar el modelo entrenado
    model.save(save_path)
    
    print(f"Modelo guardado en: {save_path}")
