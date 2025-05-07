import cv2
import os
import numpy as np
from argparse import Namespace
from tracker_wrapper import TrackerWrapper
from ultralytics import YOLO
from metrics import TrackingMetrics, IDF1Metrics, HOTAMetrics, MOTAMetrics

CLASSES = {
    0: "negra",
    1: "blanca",
    2: "verde",
    3: "azul",
    4: "negra-d",
    5: "blanca-d",
    6: "verde-d",
    7: "azul-d",
}

COLORS = {
    "negra": (0, 0, 255),
    "blanca": (0, 255, 0),
    "verde": (255, 0, 0),
    "azul": (255, 255, 0),
    "negra-d": (0, 165, 255),
    "blanca-d": (255, 165, 0),
    "verde-d": (255, 105, 180),
    "azul-d": (255, 0, 255),
}


def update_memory(tracked_objects, memory, classes) -> None:
    FRAME_AGE = 60
    PERMANENT_DEFECT_THRESHOLD = (
        3  # Número de frames consecutivos para marcar como "defecto permanente"
    )

    for obj in tracked_objects:
        track_id = int(obj[4])
        detected_class = classes[int(obj[6])]
        is_defective = detected_class.endswith("-d")

        if track_id in memory:
            entry = memory[track_id]

            if entry.get("permanent_defect", False):
                entry["visible_frames"] = FRAME_AGE
                continue

            if is_defective:
                entry["defect_counter"] = entry.get("defect_counter", 0) + 1
            else:
                entry["defect_counter"] = 0

            if entry["defect_counter"] >= PERMANENT_DEFECT_THRESHOLD:
                entry["permanent_defect"] = True
                entry["defective"] = True
                detected_class = detected_class

            entry["defective"] = entry.get("permanent_defect", False) or is_defective
            entry["visible_frames"] = FRAME_AGE
            entry["class"] = detected_class
        else:
            memory[track_id] = {
                "defective": is_defective,
                "visible_frames": FRAME_AGE,
                "class": detected_class,
                "defect_counter": 1 if is_defective else 0,
                "permanent_defect": False,
            }

    for track_id in list(memory):
        memory[track_id]["visible_frames"] -= 1
        if memory[track_id]["visible_frames"] <= 0:
            del memory[track_id]


def draw_detections(frame, outputs, classes, colors):
    for obj in outputs:
        x1, y1, x2, y2 = map(int, obj[:4])
        track_id = int(obj[4])
        conf = obj[5]
        detected_class = classes[int(obj[6])]
        color = colors[detected_class]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{detected_class} {track_id} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


# Configuración
video_path = "./test_640x640.mp4"
labels_path = "./labels"

# Inicializar métricas de tracking
metrics = TrackingMetrics()

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Error opening video file: {video_path}")
    exit(1)

print(f"Video has {video.get(cv2.CAP_PROP_FRAME_COUNT)} frames.")

out_ground_truth_path = "./results/ground_truth.mp4"
os.makedirs(os.path.dirname(out_ground_truth_path), exist_ok=True)
out_ground_truth = cv2.VideoWriter(
    out_ground_truth_path,
    cv2.VideoWriter_fourcc(*"avc1"),
    30,
    (int(video.get(3)), int(video.get(4))),
)

out_results_path = "./results/results.mp4"
os.makedirs(os.path.dirname(out_results_path), exist_ok=True)
out_results = cv2.VideoWriter(
    out_results_path,
    cv2.VideoWriter_fourcc(*"avc1"),
    30,
    (int(video.get(3)), int(video.get(4))),
)

model_path = "../../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU.engine"
model = YOLO(model_path, task="detect")
model(conf=0.5, half=True, imgsz=(640, 640), augment=True)

tracker_wrapper = TrackerWrapper(frame_rate=30)
frame_count = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, half=True, imgsz=(640, 640), augment=True)

    results_formatted = Namespace(
        xywh=results[0].boxes.xywh.cpu(),
        conf=results[0].boxes.conf.cpu(),
        cls=results[0].boxes.cls.cpu(),
    )

    outputs = tracker_wrapper.track(results_formatted, frame)

    memory = {}
    update_memory(outputs, memory, CLASSES)

    # Preparar detecciones para métricas
    detection_tuples = []
    for obj in outputs:
        track_id = int(obj[4])
        x1, y1, x2, y2 = obj[0], obj[1], obj[2], obj[3]

        bbox = np.array([x1, y1, x2, y2])
        detection_tuples.append((track_id, bbox))

    # Preparar ground truth para métricas
    gt_tuples = []
    gt_frame = frame.copy()
    label_file = os.path.join(labels_path, f"frame_{frame_count:06d}.txt")

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                x, y, w, h = map(
                    float, parts[1:5]
                )  # valores normalizados en formato YOLO (x,y,w,h)
                track_id = int(parts[5])

                # Convertir de coordenadas normalizadas YOLO a píxeles (x1,y1,x2,y2)
                frame_height, frame_width = frame.shape[:2]
                x1 = int((x - w / 2) * frame_width)
                y1 = int((y - h / 2) * frame_height)
                x2 = int((x + w / 2) * frame_width)
                y2 = int((y + h / 2) * frame_height)

                bbox = np.array([x1, y1, x2, y2])
                gt_tuples.append((track_id, bbox))

    # Actualizar métricas
    metrics.update(frame_count, detection_tuples, gt_tuples)

    # Calcular métricas actuales
    current_idf1, current_hota, current_mota = metrics.compute()

    # Añadir métricas al frame
    metrics_text = [
        # IDF1 metrics
        f"IDF1: {current_idf1.idf1:.3f}",
        f"IDP: {current_idf1.idp:.3f}",
        f"IDR: {current_idf1.idr:.3f}",
        # HOTA metrics
        f"HOTA: {current_hota.hota:.3f}",
        f"DetA: {current_hota.deta:.3f}",
        f"AssA: {current_hota.assa:.3f}",
        # MOTA metrics
        f"MOTA: {current_mota.mota:.3f}",
        f"IDSW: {current_mota.idsw}",
        f"FP/FN: {current_mota.fp}/{current_mota.fn}",
    ]

    for i, text in enumerate(metrics_text):
        cv2.putText(
            gt_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

    out_ground_truth.write(gt_frame)

    results_frame = frame.copy()
    draw_detections(frame=results_frame, outputs=outputs, classes=CLASSES, colors=COLORS)
    out_results.write(results_frame)

    frame_count += 1

out_ground_truth.release()
out_results.release()
print("Ground truth and results videos saved.")

print("Frame processing completed.")

# Calcular y mostrar métricas finales
final_idf1, final_hota, final_mota = metrics.compute()

print("\nMétricas IDF1 finales:")
print(f"IDF1: {final_idf1.idf1:.3f}")
print(f"IDP: {final_idf1.idp:.3f}")
print(f"IDR: {final_idf1.idr:.3f}")
print(f"IDTP/IDFP/IDFN: {final_idf1.idtp}/{final_idf1.idfp}/{final_idf1.idfn}")

print("\nMétricas HOTA finales:")
print(f"HOTA: {final_hota.hota:.3f}")
print(f"DetA: {final_hota.deta:.3f}")
print(f"AssA: {final_hota.assa:.3f}")
print(f"TP/FP/FN: {final_hota.tp}/{final_hota.fp}/{final_hota.fn}")

print("\nMétricas MOTA finales:")
print(f"MOTA: {final_mota.mota:.3f}")
print(f"IDSW: {final_mota.idsw}")
print(f"FP/FN: {final_mota.fp}/{final_mota.fn}")
print(f"GT total: {final_mota.gt_total}")

# Guardar métricas en archivo
results_file = "./results/metrics.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, "w") as f:
    # Guardar métricas IDF1
    f.write("Métricas IDF1:\n")
    f.write(f"IDF1: {final_idf1.idf1:.3f}\n")
    f.write(f"IDP: {final_idf1.idp:.3f}\n")
    f.write(f"IDR: {final_idf1.idr:.3f}\n")
    f.write(f"IDTP/IDFP/IDFN: {final_idf1.idtp}/{final_idf1.idfp}/{final_idf1.idfn}\n\n")

    # Guardar métricas HOTA
    f.write("Métricas HOTA:\n")
    f.write(f"HOTA: {final_hota.hota:.3f}\n")
    f.write(f"DetA: {final_hota.deta:.3f}\n")
    f.write(f"AssA: {final_hota.assa:.3f}\n")
    f.write(f"TP/FP/FN: {final_hota.tp}/{final_hota.fp}/{final_hota.fn}\n\n")

    # Guardar métricas MOTA
    f.write("Métricas MOTA:\n")
    f.write(f"MOTA: {final_mota.mota:.3f}\n")
    f.write(f"IDSW: {final_mota.idsw}\n")
    f.write(f"FP/FN: {final_mota.fp}/{final_mota.fn}\n")
    f.write(f"GT total: {final_mota.gt_total}\n")

video.release()
cv2.destroyAllWindows()
