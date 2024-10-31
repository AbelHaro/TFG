from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11m.pt"
video_path = "../datasets_labeled/prueba/video_movimiento.mp4"
output_dir = "../inference_predictions"
output_video_path = "../inference_predictions/tracking_lines/video_tracking_yolo11m_FP32.mp4"  # Ruta de salida del video

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

# Obtener dimensiones del video para el VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el codificador y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Para formato MP4
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)

        # Verificar si hay resultados y si hay detecciones
        if results and len(results) > 0:
            first_result = results[0]

            # Verificar si first_result tiene cajas
            if first_result.boxes is not None:
                boxes = first_result.boxes.xywh.cpu()
                # Verificar si hay IDs en las cajas
                if hasattr(first_result.boxes, 'id') and first_result.boxes.id is not None:
                    track_ids = first_result.boxes.id.int().cpu().tolist()
                    annotated_frame = first_result.plot()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        
                        # Ajustar el grosor de la línea a 2 para que sea más delgada
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                    cv2.imshow("YOLO11 Tracking", annotated_frame)

                    # Escribir el frame anotado en el video de salida
                    out.write(annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        else:
            print("No detecciones en este fotograma.")
    else:
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
