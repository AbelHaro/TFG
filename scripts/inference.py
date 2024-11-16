from ultralytics import YOLO
import torch
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

model_path = "../models/canicas/2024_11_15/2024_11_15_canicas_yolo11m.pt"
video_path = "../datasets_labeled/videos/video_con_defectos.mp4"
output_dir = "../inference_predictions"

model = YOLO(model_path)

results = model.predict(task='detect', source=video_path, save=True, project=output_dir, half=False, conf=0.2, device=0, show=False)        