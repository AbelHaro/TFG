import cv2
import os
import datetime
import signal
import sys
import numpy as np

# Crear el directorio ./videos si no existe
os.makedirs("./videos", exist_ok=True)

# Inicializar la cámara (2 es el índice de tu cámara, cámbialo si hace falta)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    sys.exit(1)

# Configurar la cámara para Full HD
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# Capturamos un frame para comprobar resolución
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo capturar el primer fotograma.")
    cap.release()
    sys.exit(1)

height_native, width_native = frame.shape[:2]
print(f"Resolución de la cámara: {width_native}x{height_native}")

# Asegurarnos de que el frame está en formato BGR
if len(frame.shape) < 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

# Nombres de los archivos de salida (ahora con extensión .mp4)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename_full_hd = f"./videos/{timestamp}_1920x1080.mp4"  # Cambiado a .mp4
filename_square = f"./videos/{timestamp}_1080x1080.mp4"   # Cambiado a .mp4

# Configurar VideoWriters con codec MP4V (para formato .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Cambiado a 'mp4v'
FPS = 30.0

# Crear los writers con dimensiones específicas
out_full_hd = cv2.VideoWriter(
    filename_full_hd,
    fourcc,
    FPS,
    (1920, 1080),
    isColor=True
)

out_square = cv2.VideoWriter(
    filename_square,
    fourcc,
    FPS,
    (1080, 1080),
    isColor=True
)

if not out_full_hd.isOpened() or not out_square.isOpened():
    print("Error: No se pudo crear alguno de los archivos de video.")
    cap.release()
    sys.exit(1)

def crop_center(frame, crop_width, crop_height):
    h, w = frame.shape[:2]
    start_x = w // 2 - crop_width // 2
    start_y = h // 2 - crop_height // 2
    return frame[start_y:start_y+crop_height, start_x:start_x+crop_width]

def cleanup(sig=None, frame=None):
    print("\nCerrando programa...")
    if cap.isOpened():
        cap.release()
    if out_full_hd.isOpened():
        out_full_hd.release()
    if out_square.isOpened():
        out_square.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

print("Grabando videos en 1920x1080 y 1080x1080... Presiona Ctrl+C para detener.")
print("Los videos se guardarán en formato MP4 con codec MP4V")
print("Puedes reproducirlos con: ffplay [nombre_del_video].mp4")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el fotograma.")
            break

        # Asegurarnos de que el frame está en BGR
        if len(frame.shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # Si es RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Verificar y ajustar dimensiones si es necesario
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))

        # Guardar el frame completo para Full HD
        out_full_hd.write(frame)

        # Recortar y guardar el frame para formato cuadrado
        square_frame = crop_center(frame, 1080, 1080)
        out_square.write(square_frame)

        # Mostrar ambos frames (redimensionados para mejor visualización)
        cv2.imshow("Full HD (1920x1080)", cv2.resize(frame, (960, 540)))
        cv2.imshow("Cuadrado (1080x1080)", cv2.resize(square_frame, (540, 540)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error durante la grabación: {str(e)}")
finally:
    cleanup()