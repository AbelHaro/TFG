from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n_FP16.engine"
video_path = "../datasets_labeled/prueba/video_movimiento.mp4"
output_video_path = "../inference_predictions/tracking_lines/video_tracking_yolo11n_FP16.mp4"

model = YOLO(model_path, task="detect")

cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

# Obtener FPS del video original
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)

    if results and len(results) > 0:
        first_result = results[0]

        if first_result.boxes is not None:
            boxes = first_result.boxes.xywh.cpu()
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
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                out.write(annotated_frame)
            else:
                # Mantener el estado de seguimiento si no hay detecciones
                for track_id, track in track_history.items():
                    if track:
                        last_point = track[-1]
                        # Repetir el último punto para el frame actual
                        cv2.circle(annotated_frame, (int(last_point[0]), int(last_point[1])), 5, (0, 0, 255), -1)
                        track.append(last_point)
                out.write(annotated_frame)
        else:
            print("No detecciones en este fotograma.")
    else:
        print("No hay resultados de detección.")

cap.release()
out.release()
cv2.destroyAllWindows()
