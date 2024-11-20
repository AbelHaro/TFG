from ultralytics import YOLO
from check_gpu_exists import exists_gpu
import cv2

if not exists_gpu():
    exit()

model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine'
video_path = '../datasets_labeled/videos/video_general_defectos_3.mp4'
output_dir = "../inference_predictions"

model = YOLO(model_path)

t1 = cv2.getTickCount()
results = model.predict(task='detect', source=video_path, save=True, project=output_dir, half=False, conf=0.5, device=0, show=False)
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print(f"Tiempo total de procesamiento: {time:.3f} segundos")