from ultralytics import YOLO
import cv2
import torch
from check_gpu_exists import exists_gpu

if not exists_gpu():
    exit()

# Cargar el modelo
model_path = "../models/canicas/2024_10_21/2024_10_21_canicas_yolo11x.pt"
model = YOLO(model_path)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir un frame de la cámara.")
        break

    # Redimensionar el frame a 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Realizar la inferencia directamente en el frame redimensionado
    results = model.predict(source=frame_resized, save=False, half=False, conf=0.5, device=0)

    # Dibujar las detecciones en el frame original (redimensionado)
    annotated_frame = results[0].plot()

    # Mostrar el frame con las detecciones
    cv2.imshow("Camera Feed", annotated_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
