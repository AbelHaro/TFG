import cv2
import os
import subprocess
import argparse
import time
from argparse import Namespace
from tracker_wrapper import TrackerWrapper
from ultralytics import YOLO
import numpy as np
import sys

# Añadir el directorio de scripts al path para poder importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.inference.lib.hardware_stats_usage import create_tegrastats_file

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
    x1, y1, x2, y2 = map(int, obj[:4])

    if class_name is not None and colors is not None:
        if memory is not None and track_id in memory:
            entry = memory[track_id]
            if entry.get("permanent_defect", False):
                class_name = entry["class"]

        color = colors[class_name]
        label_text = f"ID:{track_id} {class_name}"
        if conf is not None:
            label_text += f" {conf:.2f}"
    else:
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
    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
        if classes_ids is not None and classes_dict is not None and colors_dict is not None:
            class_name = classes_dict[int(classes_ids[i])]
            conf = confidences[i] if confidences is not None else None
            draw_detection(frame, box, track_id, conf, class_name, memory, colors_dict)
        else:
            draw_detection(frame, box, track_id)


def main():
    parser = argparse.ArgumentParser(description="Secuencial tracking with camera FPS simulation")
    parser.add_argument(
        "--video",
        type=str,
        default="../../datasets_labeled/videos/contar_objetos_88_2min.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU.engine",
        help="Path to YOLO model file",
    )
    parser.add_argument(
        "--fps", type=str, default="30", help="Camera FPS to simulate ('infinite' for no limit)"
    )
    args = parser.parse_args()

    # Configuración de FPS
    if args.fps.lower() == "infinite":
        frame_time = 0  # No hay límite de tiempo
    else:
        try:
            fps = float(args.fps)
            frame_time = 1.0 / fps
        except ValueError:
            print(f"Error: FPS debe ser un número o 'infinite', no '{args.fps}'")
            sys.exit(1)

    # Configuración de video
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print(f"Error opening video file: {args.video}")
        exit(1)

    print(f"Video has {video.get(cv2.CAP_PROP_FRAME_COUNT)} frames.")

    out_results_path = "./results/tracking_results.mp4"
    os.makedirs(os.path.dirname(out_results_path), exist_ok=True)
    out_results = cv2.VideoWriter(
        out_results_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (int(video.get(3)), int(video.get(4))),
    )

    memory = {}

    model = YOLO(args.model, task="detect")
    model(conf=0.5, half=True, imgsz=(640, 640), augment=True)

    tracker_wrapper = TrackerWrapper(frame_rate=30)
    frame_count = 0
    frames_dropped = 0

    # Configuración para tegrastats
    tegra_stats_output = "./results/hardware_usage.txt"
    os.makedirs(os.path.dirname(tegra_stats_output), exist_ok=True)

    # Iniciar tegrastats
    print("Starting tegrastats...")
    tegrastats_process = subprocess.Popen(
        ["tegrastats", "--interval", "100", "--logfile", tegra_stats_output],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    dumy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dumy_frame, conf=0.5, half=True, imgsz=(640, 640), augment=True)

    t1 = cv2.getTickCount()
    last_frame_time = time.time()
    print(f"Starting tracking with{'out' if frame_time == 0 else ''} FPS limit ({args.fps} FPS)...")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        current_time = time.time()
        processing_start = current_time

        results = model.predict(frame, conf=0.5, half=True, imgsz=(640, 640), augment=True)

        results_formatted = Namespace(
            xywh=results[0].boxes.xywh.cpu(),
            conf=results[0].boxes.conf.cpu(),
            cls=results[0].boxes.cls.cpu(),
        )

        outputs = tracker_wrapper.track(results_formatted, frame)

        update_memory(outputs, memory, CLASSES)

        # Create and annotate results frame
        boxes = [obj[:4] for obj in outputs]
        track_ids = [int(obj[4]) for obj in outputs]
        confidences = [obj[5] for obj in outputs]
        classes_ids = [int(obj[6]) for obj in outputs]

        annotate_frame(
            frame=frame,
            boxes=boxes,
            track_ids=track_ids,
            classes_ids=classes_ids,
            confidences=confidences,
            classes_dict=CLASSES,
            colors_dict=COLORS,
            memory=memory,
        )
        out_results.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end="\r")

        # Control de FPS después de procesar el frame
        if frame_time > 0:
            processing_time = time.time() - processing_start
            if processing_time > frame_time:
                frames_to_skip = int(processing_time / frame_time)
                for _ in range(frames_to_skip):
                    ret, _ = video.read()
                    if not ret:
                        break
                    frames_dropped += 1
                    print(f"Skipped frame {frame_count} to maintain timing...", end="\r")
                continue
            else:
                sleep_time = frame_time - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    t2 = cv2.getTickCount()
    total_time = (t2 - t1) / cv2.getTickFrequency()

    print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds.")
    print(f"Dropped {frames_dropped} frames to maintain timing.")
    print(f"Effective FPS: {frame_count / total_time:.2f}")

    # Detener tegrastats
    print("Stopping tegrastats...")
    tegrastats_process.terminate()
    tegrastats_process.wait()

    # Procesar datos de tegrastats
    print("Processing hardware usage data...")
    hardware_usage_csv = "./results/hardware_usage.csv"
    create_tegrastats_file(tegra_stats_output, hardware_usage_csv, total_time)

    out_results.release()
    video.release()
    cv2.destroyAllWindows()

    print(f"Tracking completed. Results saved to {out_results_path}")
    print(f"Hardware usage data saved to {hardware_usage_csv}")


if __name__ == "__main__":
    main()
