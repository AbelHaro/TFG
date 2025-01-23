from ultralytics import YOLO
#from ..check_gpu_exists import exists_gpu
import cv2
import os

#if not exists_gpu():
#    exit()

model_path = '/TFG/models/canicas/2024_11_28/out.engine'
video_path = '/TFG/datasets_labeled/2024_11_28_canicas_dataset/test/images'
output_dir = "../inference_predictions"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = YOLO(model_path)


t1 = cv2.getTickCount()
results = model.predict(task='detect', source=video_path, save=True, project=output_dir, half=False, conf=0.5, device=0, show=False)
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print(f"Tiempo total de procesamiento: {time:.3f} segundos")