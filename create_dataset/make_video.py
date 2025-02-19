import cv2
import os
import datetime
import signal
import sys

# Crear el directorio ./videos si no existe
os.makedirs("./videos", exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Obtener la fecha y hora actual para el nombre del archivo
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"./videos/{timestamp}.mp4"

# Definir el códec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
