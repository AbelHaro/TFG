import cv2
import os
import datetime
import signal
import sys

# Crear el directorio ./videos_nuevo si no existe
os.makedirs("./videos_nuevo", exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Definir resolución
width = 640
height = 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Configurar el códec y formato del video
codec = 'MJPG'  # Usando Motion JPEG
format = 'avi'

# Obtener la fecha y hora actual para el nombre del archivo
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"./videos_nuevo/video_original.{format}"

# Definir el códec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

def cleanup(signal, frame):
    print("\nCerrando programa...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

print("Grabando video... Presiona Ctrl+C para detener.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break
    
    # Verificar y ajustar el formato de píxel si es necesario
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB si es necesario

    # Escribir el fotograma en el archivo de video
    out.write(frame)

    # Mostrar la imagen en una ventana
    cv2.imshow("Grabando", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()
