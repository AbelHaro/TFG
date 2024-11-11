import cv2

from ultralytics import solutions


model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n_FP16.engine"
video_path = "../datasets_labeled/prueba/video_movimiento.mp4"
output_dir = "../inference_predictions"
output_video_path = "../inference_predictions/distance/video_distance_yolo11n_FP16.mp4"  # Ruta de salida del video

cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Aumentar el tamaño del rectángulo
# Ejemplo: aumentar el ancho y la altura
width_increase = 50  # Aumentar 50 píxeles en ancho
height_increase = 20  # Aumentar 20 píxeles en altura

# Nueva definición del rectángulo
speed_region = [
    (20, 400),                  # Esquina inferior izquierda
    (1080 + width_increase, 404),  # Esquina inferior derecha
    (1080 + width_increase, 360 - height_increase),  # Esquina superior derecha
    (20, 360 - height_increase)      # Esquina superior izquierda
]

speed = solutions.SpeedEstimator(model=model_path, region=speed_region, show=True)

while cap.isOpened():
    success, im0 = cap.read()

    if success:
        out = speed.estimate_speed(im0)
        video_writer.write(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
cv2.destroyAllWindows()