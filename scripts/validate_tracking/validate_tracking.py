import cv2
import os
import numpy as np
from argparse import Namespace
from tracker_wrapper import TrackerWrapper
from ultralytics import YOLO
from metrics import TrackingMetrics, IDF1Metrics, HOTAMetrics, MOTAMetrics, MOTPMetrics

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


def draw_detection(frame, obj, track_id, conf=None, class_name=None, memory=None, colors=None):
    """
    Draw a single detection on the frame.
    """
    x1, y1, x2, y2 = map(int, obj[:4])

    if class_name is not None and colors is not None:
        # Check memory for the object state if available
        if memory is not None and track_id in memory:
            entry = memory[track_id]
            if entry.get("permanent_defect", False):
                class_name = entry["class"]

        color = colors[class_name]
        label_text = f"ID:{track_id} {class_name}"
        if conf is not None:
            label_text += f" {conf:.2f}"
    else:
        # Ground truth case
        color = (0, 255, 0)
        label_text = f"ID: {track_id}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def annotate_frame(
    frame,
    boxes,
    track_ids,
    classes_ids=None,
    confidences=None,
    classes_dict=None,
    colors_dict=None,
    memory=None,
):
    """
    Annotate a frame with bounding boxes and labels.

    Args:
        frame: The frame to annotate
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        track_ids: List of track IDs corresponding to each box
        classes_ids: Optional list of class IDs for each box
        confidences: Optional list of confidence scores for each box
        classes_dict: Dictionary mapping class IDs to class names
        colors_dict: Dictionary mapping class names to colors
        memory: Optional memory dictionary for tracking object states
    """
    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
        if classes_ids is not None and classes_dict is not None and colors_dict is not None:
            class_name = classes_dict[int(classes_ids[i])]
            conf = confidences[i] if confidences is not None else None
            draw_detection(frame, box, track_id, conf, class_name, memory, colors_dict)
        else:
            # Ground truth annotations (simple version without memory)
            draw_detection(frame, box, track_id)


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
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(video.get(3)), int(video.get(4))),
)

out_results_path = "./results/results.mp4"
os.makedirs(os.path.dirname(out_results_path), exist_ok=True)
out_results = cv2.VideoWriter(
    out_results_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(video.get(3)), int(video.get(4))),
)

memory = {}

model_path = "../../models/canicas/2025_02_24/2025_02_24_canicas_yolo11m_FP16_GPU.engine"
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

    gt_boxes = []
    gt_track_ids = []
    gt_class_ids = []

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                track_id = int(parts[5])

                # Convert normalized YOLO coordinates to pixels
                frame_height, frame_width = frame.shape[:2]
                x1 = int((x - w / 2) * frame_width)
                y1 = int((y - h / 2) * frame_height)
                x2 = int((x + w / 2) * frame_width)
                y2 = int((y + h / 2) * frame_height)

                bbox = np.array([x1, y1, x2, y2])
                gt_tuples.append((track_id, bbox))

                gt_boxes.append(bbox)
                gt_track_ids.append(track_id)
                gt_class_ids.append(class_id)

    # Annotate ground truth frame with boxes
    annotate_frame(
        frame=gt_frame,
        boxes=gt_boxes,
        track_ids=gt_track_ids,
        classes_ids=gt_class_ids,
        classes_dict=CLASSES,
        colors_dict=COLORS,
    )

    # Actualizar métricas
    metrics.update(frame_count, detection_tuples, gt_tuples)

    # Calcular métricas actuales
    current_idf1, current_hota, current_mota, current_motp = metrics.compute()

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
        # MOTP metrics
        f"MOTP: {current_motp.motp:.3f}",
    ]

    for i, text in enumerate(metrics_text):
        cv2.putText(
            gt_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

    # Write ground truth frame with metrics
    out_ground_truth.write(gt_frame)

    # Create and annotate results frame
    results_frame = frame.copy()
    boxes = [obj[:4] for obj in outputs]
    track_ids = [int(obj[4]) for obj in outputs]
    confidences = [obj[5] for obj in outputs]
    classes_ids = [int(obj[6]) for obj in outputs]

    annotate_frame(
        frame=results_frame,
        boxes=boxes,
        track_ids=track_ids,
        classes_ids=classes_ids,
        confidences=confidences,
        classes_dict=CLASSES,
        colors_dict=COLORS,
        memory=memory,  # Pass memory to annotate_frame
    )
    out_results.write(results_frame)

    frame_count += 1

out_ground_truth.release()
out_results.release()
print("Ground truth and results videos saved.")

print("Frame processing completed.")

# Calcular y mostrar métricas finales
final_idf1, final_hota, final_mota, final_motp = metrics.compute()

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

print("\nMétricas MOTP finales:")
print(f"MOTP: {final_motp.motp:.3f}")
print(f"Total distance: {final_motp.total_distance:.3f}")
print(f"Total matches: {final_motp.total_matches}")

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
    f.write(f"GT total: {final_mota.gt_total}\n\n")

    # Guardar métricas MOTP
    f.write("Métricas MOTP:\n")
    f.write(f"MOTP: {final_motp.motp:.3f}\n")
    f.write(f"Total distance: {final_motp.total_distance:.3f}\n")
    f.write(f"Total matches: {final_motp.total_matches}\n")

video.release()
cv2.destroyAllWindows()
