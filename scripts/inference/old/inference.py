from ultralytics import YOLO
import cv2

model_path = '/TFG/models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16_GPU.engine'
video_path = f'../../datasets_labeled/videos/contar_objetos_40_2min.mp4'

model = YOLO(model_path)
model(conf=0.5, device="cuda:0")
print("[INFO] Model loaded")

cap = cv2.VideoCapture(video_path)

print("[INFO] Starting inference")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video finished")
        break

    preprocessed = model.predictor.preprocess([frame])

    output = model.predictor.inference(preprocessed)

    results = model.predictor.postprocess(output, preprocessed, [frame])

    print(f"[INFO] {results[0].boxes}")
