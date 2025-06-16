import cv2
import os
import datetime
import signal
import sys

# Crear carpeta de videos
os.makedirs("./videos", exist_ok=True)

# Inicializar cámara
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    sys.exit(1)

# Configurar resolución 640x480 y MJPG para mejor compatibilidad
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verificar frame inicial
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo capturar el frame inicial.")
    cap.release()
    sys.exit(1)

if frame.shape[1] != 640 or frame.shape[0] != 480:
    print(
        f"Advertencia: la cámara devuelve {frame.shape[1]}x{frame.shape[0]}, no 640x480."
    )

# Crear nombre del archivo
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"./videos/{timestamp}_640x640.mp4"

# Crear VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 640))
if not out.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 640))
    if not out.isOpened():
        print("Error: no se pudo crear el archivo de video.")
        cap.release()
        sys.exit(1)


# Manejo de cierre con Ctrl+C
def cleanup(sig=None, frame=None):
    print("\nCerrando...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)

print("Grabando video (640x480 → 640x640). Presiona Ctrl+C o 'q' para detener.")
print(f"Archivo de salida: {filename}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el fotograma.")
            break

        # Redimensionar a 640x640
        resized = cv2.resize(frame, (640, 640))

        out.write(resized)
        cv2.imshow("Grabando", resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("Error durante la grabación:", e)
finally:
    cleanup()
