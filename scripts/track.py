from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "../models/canicas/2024_11_15/2024_11_15_canicas_yolo11m.pt"
video_path = "../datasets_labeled/videos/video_con_defectos.mp4"
output_dir = "../inference_predictions"

model = YOLO(model_path)

t1 = cv2.getTickCount()
results = model.track(task="detect",source=video_path, save=True, project=output_dir, half=False, conf=0.5, device=0, show=False, tracker="bytetrack.yaml")
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print(f"Tiempo total de procesamiento: {time:.3f} segundos")