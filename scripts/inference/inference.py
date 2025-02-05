from ultralytics import YOLO
import cv2

model_path = '/TFG/models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16_GPU.engine'
video_path = f'../../datasets_labeled/videos/contar_objetos_40_2min.mp4'

model = YOLO(model_path)


cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=video_path, conf=0.5, device="cuda:0")
    
