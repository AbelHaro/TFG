import cv2
import os
import datetime
import signal
import sys

# Crear el directorio ./images si no existe
os.makedirs("./images", exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

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
        filename = f"./images/{timestamp}.jpg"
        
        # Guardar la imagen
        cv2.imwrite(filename, frame)
        print(f"Foto guardada: {filename}")
    
    elif key == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
