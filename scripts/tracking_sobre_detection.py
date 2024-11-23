from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace

# Ruta al modelo YOLO
model_path = "../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine"
video_path = '../datasets_labeled/videos/video_general_defectos_3.mp4'
output_dir = "../inference_predictions/probando_tracking"

# Cargar el modelo YOLO
model = YOLO(model_path)

# Configuraciones del tracker
args = Namespace(
    tracker_type='bytetrack',         # Tipo de tracker
    track_high_thresh=0.25,          # Umbral para la primera asociación
    track_low_thresh=0.1,            # Umbral para la segunda asociación
    new_track_thresh=0.25,           # Umbral para iniciar un nuevo track
    track_buffer=30,                 # Buffer para determinar cuándo eliminar tracks
    match_thresh=0.8,                # Umbral para emparejar tracks
    fuse_score=True                  # Fusionar la confianza con la distancia IoU antes del emparejamiento
)

# Inicializar el tracker
tracker = BYTETracker(args, frame_rate=30)

cap = cv2.VideoCapture(video_path)

# Inicializar un diccionario para almacenar el conteo de objetos por clase
object_counts = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predicción con el modelo YOLO
    results = model.predict(task="detect", source=frame, conf=0.5, device=0, show=False)
    
    for result in results:
        # Extraer las detecciones en formato numpy
        boxes = result.boxes.xywh.cpu().numpy()  # Coordenadas (x, y, w, h)
        confidences = result.boxes.conf.cpu().numpy()  # Confianza de la predicción
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs de las clases

        #if len(boxes) == 0:
        #    continue

        # Crear un objeto que tenga los atributos necesarios
        class Detections:
            def __init__(self, boxes, confidences, class_ids):
                self.conf = confidences
                self.xywh = boxes
                self.cls = class_ids

        # Crear las detecciones en el formato esperado por BYTETracker
        detections = Detections(boxes, confidences, class_ids)

        # Actualizar el tracker con las detecciones
        outputs = tracker.update(detections, frame)

    
        # Contar objetos rastreados por clase
        for track in outputs:
            print(track)
            class_id = int(track[6])  # Clase rastreada
            object_counts[class_id] += 1  # Incrementar el conteo

print(f"La longitud de outputs es: {len(outputs)}, la longitud de results es: {len(results)}")
# Imprimir el conteo final de objetos por clase
print("Conteo de objetos por clase:", dict(object_counts))
