from ultralytics import YOLO
import torch
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

model_path = "../models/canicas/2024_10_21/2024_10_21_canicas_yolo11x.pt"
video_path = "../datasets_labeled/prueba/video.webm"
output_dir = "../inference_predictions"

model = YOLO(model_path)

results = model.track(source=video_path, save=True, save_frames=True, project=output_dir, half=False, conf=0.5, device=0, show=False, tracker="bytetrack.yaml")    