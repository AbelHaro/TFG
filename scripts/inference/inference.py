from ultralytics import YOLO
import cv2

model_path = '/TFG/models/canicas/2024_11_28/2024_11_28_canicas_yolo11n.pt'
video_path = f'../../datasets_labeled/videos/contar_objetos_40_2min.mp4'

model = YOLO("yolo11n")


t1 = cv2.getTickCount()
results = model.predict(source=video_path, conf=0.5, device="cuda:0")
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print(f"Tiempo total de procesamiento: {time:.3f} segundos")
