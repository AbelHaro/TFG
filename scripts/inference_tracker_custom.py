import cv2
import numpy as np
from ultralytics import YOLO
import os

model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.pt'
video_path = '../datasets_labeled/videos/video_con_defectos.mp4'
output_dir = '../inference_predictions/custom_tracker'  # Directorio de salida

# Asegurarse de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Nombre del archivo de salida
output_video_path = os.path.join(output_dir, 'video_con_tracking.mp4')

model = YOLO(model_path)

# Definir clases y estructura de memoria
clases = {
    0: 'negra',
    1: 'blanca',
    2: 'verde',
    3: 'azul',
    4: 'negra-d',
    5: 'blanca-d',
    6: 'verde-d',
    7: 'azul-d',
}
memory = {}

# Función para actualizar la memoria
def actualizar_memoria(track_id, clase_detectada):
    if track_id not in memory:
        memory[track_id] = {'defectuosa': clase_detectada.endswith('-d'), 'frames_visibles': 30}
    else:
        # Marca como defectuosa de forma permanente si tiene el sufijo "-d"
        memory[track_id]['defectuosa'] |= clase_detectada.endswith('-d')
        memory[track_id]['frames_visibles'] = 30  # Reinicia el contador de frames

# Procesar video
cap = cv2.VideoCapture(video_path)  # O usa 0 para la cámara en tiempo real
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar el escritor de video para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usar el códec mp4v
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar la imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(
        source=frame,           # (str, optional) source directory for images or videos
        device=0,                    # (int, optional) GPU id (0-9) or -1 for CPU
        persist=True,
        tracker='bytetrack.yaml',    # (str, optional) filename of tracker YAML
    )
    
    if(results[0].boxes.id == None):
        print("No hay ids")
        continue
    # Obtener coordenadas de las detecciones y IDs
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Actualizar la memoria con las detecciones y dibujar los resultados
    for box, obj_id, cls in zip(boxes, ids, classes):
        xmin, ymin, xmax, ymax = box
        clase_detectada = clases[cls]
        
        # Actualizar la memoria de acuerdo a si la clase es defectuosa
        actualizar_memoria(obj_id, clase_detectada)

        # Determinar el estado de la canica (defectuosa o no)
        estado = 'Defectuosa' if memory[obj_id]['defectuosa'] else 'Sin defecto'
        color = (0, 0, 255) if memory[obj_id]['defectuosa'] else (0, 255, 0)

        # Dibujar rectángulo alrededor del objeto detectado
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Agregar texto con el ID y estado
        cv2.putText(frame, f'{obj_id} - {clase_detectada}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Escribir el frame procesado al video de salida
    out.write(frame)

    # Mostrar el resultado
    #cv2.imshow('Detección y Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Liberar el escritor de video
cv2.destroyAllWindows()
