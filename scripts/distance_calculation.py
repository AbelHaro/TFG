import cv2

from ultralytics import solutions


model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n_FP16.engine"
video_path = "../datasets_labeled/prueba/video_movimiento.mp4"
output_dir = "../inference_predictions"
output_video_path = "../inference_predictions/distance/video_distance_yolo11n_FP16.mp4"  # Ruta de salida del video


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
distance = solutions.DistanceCalculation(model=model_path, show=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = distance.calculate(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()