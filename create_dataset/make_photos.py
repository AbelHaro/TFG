import cv2
import os
import datetime
import signal
import sys
import uuid

# Crear los directorios para las diferentes resoluciones
os.makedirs("./images/1080", exist_ok=True)
os.makedirs("./images/640", exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'f' para tomar una foto o 'q' para salir.")


def cleanup(signal, frame):
    print("\nCerrando programa...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    # Mostrar la imagen en una ventana
    cv2.imshow("Captura", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):
        # Obtener la fecha y hora actual
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        unique_id = uuid.uuid4()
        base_filename = f"{timestamp}-{unique_id}.jpg"

        # Redimensionar la imagen a 1080 x 1080
        frame_1080 = cv2.resize(frame, (1080, 1080))
        
        # Crear versión 640 x 640
        frame_640 = cv2.resize(frame_1080, (640, 640))

        # Guardar ambas versiones
        cv2.imwrite(f"./images/1080/{base_filename}", frame_1080)
        cv2.imwrite(f"./images/640/{base_filename}", frame_640)
        print(f"Fotos guardadas en 1080p y 640p: {base_filename}")

    elif key == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
