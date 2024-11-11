from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11m.pt"
video_path = "../datasets_labeled/prueba/video_movimiento.mp4"
output_dir = "../inference_predictions"

model = YOLO(model_path)

results = model.track(task="detect",source=video_path, save=True, project=output_dir, half=False, conf=0.5, device=0, show=False, tracker="bytetrack.yaml")
    