import cv2  # type: ignore
import os
import torch.multiprocessing as mp  # type: ignore
from argparse import Namespace
import numpy as np
import argparse
from tcp import tcp_server, handle_send
from shared_circular_buffer import SharedCircularBuffer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_objects",
    default=40,
    type=int,
    choices=[0, 18, 40, 48, 60, 70, 88, 176],
    help="Número de objetos a contar, posibles valores: {0, 18, 40, 48, 60, 70, 88, 176}, default=40",
)
parser.add_argument(
    "--model_size",
    default="n",
    type=str,
    choices=["n", "s", "m", "l", "x"],
    help="Talla del modelo {n, s, m, l, x}, default=n",
)
parser.add_argument(
    "--precision",
    default="FP16",
    type=str,
    choices=["FP32", "FP16", "INT8"],
    help="Precisión del modelo {FP32, FP16, INT8}, default=FP16",
)
parser.add_argument(
    "--hardware",
    default="GPU",
    type=str,
    choices=["GPU", "DLA0", "DLA1"],
    help="Hardware a usar {GPU, DLA0, DLA1}, default=GPU",
)
parser.add_argument(
    "--mode",
    required=True,
    default="MAXN",
    type=str,
    choices=["MAXN", "30W", "15W", "10W"],
    help="Modo de energía a usar {MAXN, 30W, 15W, 10W}, default=MAXN",
)
parser.add_argument("--tcp", default=False, type=bool, help="Usar conexión TCP, default=False")

args = parser.parse_args()

FRAME_AGE = 15


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def update_memory(tracked_objects, memory, classes):
    global FRAME_AGE

    for obj in tracked_objects:
        track_id = int(obj[4])
        detected_class = classes[int(obj[6])]

        is_defective = detected_class.endswith("-d")
        if track_id in memory:
            entry = memory[track_id]
            entry["defective"] |= is_defective
            entry["visible_frames"] = FRAME_AGE
            if entry["defective"] and not is_defective:
                detected_class += "-d"
            entry["class"] = detected_class
        else:
            memory[track_id] = {
                "defective": is_defective,
                "visible_frames": FRAME_AGE,
                "class": detected_class,
            }

    for track_id in list(memory):
        memory[track_id]["visible_frames"] -= 1
        if memory[track_id]["visible_frames"] <= 0:
            del memory[track_id]


def capture_frames(video_path, frame_queue, stop_event, tcp_conn, is_tcp):

    print(f"[DEBUG] Iniciando captura de frames")

    if not os.path.exists(video_path):
        frame_queue.put(None)
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    # Establece la resolución de la cámara a 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        frame_queue.put(None)
        raise IOError(f"Error al abrir el archivo de video: {video_path}")

    tcp_conn.wait() if is_tcp else None

    frame_count = 0

    while cap.isOpened() and not stop_event.is_set():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()

        if not ret or cv2.waitKey(1) & 0xFF == ord("q"):
            print("[PROGRAM - CAPTURE FRAMES] No se pudo leer el frame, añadiendo None a la cola")
            print("[PROGRAM - CAPTURE FRAMES - DEBUG] Se han procesado", frame_count, "frames")
            break

        t2 = cv2.getTickCount()
        total_frame_time = (t2 - t1) / cv2.getTickFrequency()
        times = {"capture": total_frame_time}
        # print(f"[DEBUG] Poniendo frame a la cola", frame.shape)
        frame_queue.put((frame, times))
        frame_count += 1

    cap.release()
    print("[PROGRAM - CAPTURE FRAMES] Video terminado, añadiendo None a la cola")
    frame_queue.put(None)

    while not stop_event.is_set():
        pass


def process_frames(frame_queue, detection_queue, model_path, stop_event, t1_start):
    from ultralytics import YOLO  # type: ignore

    times_detect_function = {}

    model = YOLO(model_path, task="detect")

    model(device="cuda:0", conf=0.5, half=True, imgsz=(640, 640), augment=True)

    # dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')

    t1_start.set()

    while True:
        item = frame_queue.get()
        # print(f"[DEBUG] Item recibido: {item}")
        if item is None:
            detection_queue.put(None)
            break

        frame, times = item

        t1 = cv2.getTickCount()

        # Preprocesa el frame
        t1_aux = cv2.getTickCount()
        preprocessed = model.predictor.preprocess([frame])
        t2_aux = cv2.getTickCount()
        times_detect_function["preprocess"] = (t2_aux - t1_aux) / cv2.getTickFrequency()

        # Realiza la inferencia
        t1_aux = cv2.getTickCount()
        output = model.predictor.inference(preprocessed)
        t2_aux = cv2.getTickCount()
        times_detect_function["inference"] = (t2_aux - t1_aux) / cv2.getTickFrequency()

        # Postprocesa los resultados
        t1_aux = cv2.getTickCount()
        results = model.predictor.postprocess(output, preprocessed, [frame])
        t2_aux = cv2.getTickCount()
        times_detect_function["postprocess"] = (t2_aux - t1_aux) / cv2.getTickFrequency()

        # results = model.predict(source=frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')

        result_formatted = Namespace(
            xywh=results[0].boxes.xywh.cpu(),
            conf=results[0].boxes.conf.cpu(),
            cls=results[0].boxes.cls.cpu(),
        )
        t2 = cv2.getTickCount()

        processing_time = (t2 - t1) / cv2.getTickFrequency()

        times["processing"] = processing_time
        times["detect_function"] = times_detect_function

        detection_queue.put((frame, result_formatted, times))

    while not stop_event.is_set():
        pass


from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore


class TrackerWrapper:
    global FRAME_AGE

    def __init__(self, frame_rate=20):
        self.args = Namespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=FRAME_AGE,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    class Detections:
        def __init__(self, boxes, confidences, class_ids):
            self.conf = confidences
            self.xywh = boxes
            self.cls = class_ids

    def track(self, detection_data, frame):
        detections = self.Detections(
            detection_data.xywh.numpy(),
            detection_data.conf.numpy(),
            detection_data.cls.numpy().astype(int),
        )
        return self.tracker.update(detections, frame)


def tracking_frames(detection_queue, tracking_queue, stop_event):

    tracker_wrapper = TrackerWrapper(frame_rate=20)

    while True:
        item = detection_queue.get()
        if item is None:
            tracking_queue.put(None)
            break

        t1 = cv2.getTickCount()
        frame, result, times = item

        outputs = tracker_wrapper.track(result, frame)

        t2 = cv2.getTickCount()

        tracking_time = (t2 - t1) / cv2.getTickFrequency()

        times["tracking"] = tracking_time
        times["objects_count"] = len(outputs)

        tracking_queue.put((frame, outputs, times))

    while not stop_event.is_set():
        pass

    os._exit(0)


def setup_aruco():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minCornerDistanceRate = 0.05
    return aruco_dict, parameters


def process_aruco_markers(frame, aruco_dict, parameters, marker_size_cm=3.527):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            corners_array = corner[0]
            top_left = corners_array[0]
            top_right = corners_array[1]
            side_length_px = np.linalg.norm(top_left - top_right)
            px_to_cm_ratio = marker_size_cm / side_length_px
            marker_id = ids[i][0]

            text = f"ID: {marker_id} Size: {marker_size_cm:.2f} cm"
            position = (int(top_left[0]), int(top_left[1]) - 10)

            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, px_to_cm_ratio if ids is not None else None


def update_object_positions(obj, last_positions, current_time, MAX_TRAJECTORY=30):
    obj_id = int(obj[4])
    xmin, ymin, xmax, ymax = map(int, obj[:4])
    center_x = int((xmin + xmax) // 2)
    center_y = int((ymin + ymax) // 2)

    if obj_id not in last_positions:
        last_positions[obj_id] = {
            "id": obj_id,
            "pos": [(center_x, center_y)],
            "time": [current_time],
            "speed": 0,
        }
    else:
        last_positions[obj_id]["pos"].append((center_x, center_y))
        last_positions[obj_id]["time"].append(current_time)

        if len(last_positions[obj_id]["pos"]) > 5:
            recent_positions = last_positions[obj_id]["pos"][-5:]
            recent_times = last_positions[obj_id]["time"][-5:]
            total_distance = sum(
                (
                    (recent_positions[i + 1][0] - recent_positions[i][0]) ** 2
                    + (recent_positions[i + 1][1] - recent_positions[i][1]) ** 2
                )
                ** 0.5
                for i in range(4)
            )
            total_time = recent_times[-1] - recent_times[0]
            if total_time > 0:
                last_positions[obj_id]["speed"] = total_distance / total_time

        if len(last_positions[obj_id]["pos"]) > MAX_TRAJECTORY:
            last_positions[obj_id]["pos"].pop(0)
            last_positions[obj_id]["time"].pop(0)

    return last_positions


def draw_and_write_frames(
    tracking_queue,
    times_queue,
    output_video_path,
    classes,
    memory,
    colors,
    stop_event,
    t2_start,
    tcp_conn,
    is_tcp,
):

    import threading
    import time

    FPS_COUNT = 0
    FPS_LABEL = 0
    out = None
    first_time = True
    last_positions = {}
    frame_number = 0
    max_speed = 0

    def reset_fps_counter(times_queue, stop_event):
        import time

        nonlocal FPS_COUNT, FPS_LABEL
        while not stop_event.is_set():
            times_queue.put(("fps", FPS_COUNT))
            FPS_LABEL = FPS_COUNT
            FPS_COUNT = 0
            time.sleep(1)

    aruco_dict, parameters = setup_aruco()

    if is_tcp:
        client_socket, server_socket = tcp_server("0.0.0.0", 8765)
        threading.Thread(target=handle_send, args=(client_socket, "READY"), daemon=True).start()

    tcp_conn.set() if is_tcp else None

    while True:
        item = tracking_queue.get()
        if item is None:
            break

        t1 = cv2.getTickCount()
        frame, tracked_objects, times = item

        frame, px_to_cm_ratio = process_aruco_markers(frame, aruco_dict, parameters)

        if first_time:
            first_time = False
            fps_reset_thread = threading.Thread(
                target=lambda: reset_fps_counter(times_queue, stop_event), daemon=True
            )
            fps_reset_thread.start()

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        update_memory(tracked_objects, memory, classes)
        msg_sended = False

        current_time = time.time()
        for obj in tracked_objects:
            if float(obj[5]) < 0.4:  # conf threshold
                continue

            last_positions = update_object_positions(obj, last_positions, current_time)

            # Draw object information
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
            detected_class = memory[obj_id]["class"]
            color = colors.get(detected_class, (255, 255, 255))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            speed = last_positions[obj_id]["speed"]

            speed_cms = speed * (px_to_cm_ratio if px_to_cm_ratio else 0)
            max_speed = max(speed_cms, max_speed)
            text = f"ID:{obj_id} {detected_class} {float(obj[5]):.2f} Speed:{speed_cms:.1f}cm/s"
            cv2.putText(
                frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

            if detected_class.endswith("-d") and not msg_sended and is_tcp:
                threading.Thread(
                    target=handle_send, args=(client_socket, "DETECTED_DEFECT"), daemon=True
                ).start()
                msg_sended = True

            # Draw trajectory
            points = last_positions[obj_id]["pos"]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 2)

        cv2.putText(
            frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"FPS: {FPS_LABEL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        out.write(frame)
        FPS_COUNT += 1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        t2 = cv2.getTickCount()
        times["writing"] = (t2 - t1) / cv2.getTickFrequency()

        frame_number += 1

        if frame_number % 20 == 0:
            print(
                f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado", end="\r", flush=True
            )

        times_queue.put(("times", times))

    if out:
        out.release()

    times_queue.put(None)
    print("[PROGRAM - DRAW AND WRITE] None añadido a la cola de tiempos")
    print("[PROGRAM - DRAW AND WRITE] La velocidad máxima registrada fue de", max_speed, "cm/s")

    if is_tcp:
        client_socket.close()
        server_socket.close()

    t2_start.set()
    stop_event.set()
    os._exit(0)


def write_to_csv(times_queue, output_file):
    from create_excel_multiprocesses import (
        create_csv_file,
        add_row_to_csv,
        add_fps_to_csv,
        create_excel_from_csv,
    )
    import os

    frame_count = 0

    times_name = "times_multiprocesses.csv"
    fps_name = "fps_multiprocesses.csv"

    times_excel_file = create_csv_file(file_name=times_name)
    fps_excel_file = create_csv_file(file_name=fps_name)

    while True:
        item = times_queue.get()

        if item is None:
            break

        label, data = item

        if label == "times":
            add_row_to_csv(times_excel_file, frame_count, data)
        elif label == "fps":
            add_fps_to_csv(fps_excel_file, frame_count, data)

    create_excel_from_csv(
        times_name, fps_name, output_name=f"multiprocesses_{output_file}_2min.xlsx"
    )

    print("[PROGRAM - WRITE TO CSV] None recibido, terminando proceso")

    os._exit(0)


def hardware_usage(output_file, stop_event, t1_start, tcp_conn, is_tcp):
    import subprocess
    from datetime import datetime
    from hardware_stats_usage import create_tegrastats_file

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    tegra_stats_output = f"/TFG/excels/tegrastats_outputs/{output_file}_{timestamp}.txt"
    output_excel_filename = f"/TFG/excels/hardware_stats_usage/{output_file}.csv"

    os.makedirs(os.path.dirname(tegra_stats_output), exist_ok=True)
    os.makedirs(os.path.dirname(output_excel_filename), exist_ok=True)

    # Espera inicial para sincronizar con el evento
    t1_start.wait()
    tcp_conn.wait() if is_tcp else None

    # Iniciar el proceso de tegrastats
    process = subprocess.Popen(
        ["tegrastats", "--interval", "100", "--logfile", tegra_stats_output],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("[PROGRAM - HARDWARE USAGE] Iniciando tegrastats...")

    stop_event.wait()
    process.terminate()
    process.wait()
    # finally:
    print("[PROGRAM - HARDWARE USAGE] Deteniendo el proceso tegrastats...")
    create_tegrastats_file(tegra_stats_output, output_excel_filename)
    print("[PROGRAM - HARDWARE USAGE] Proceso tegrastats detenido.")

    print("[PROGRAM - HARDWARE USAGE] Terminando proceso")
    os._exit(0)


def main():

    objects_count = args.num_objects
    model_name = "yolo11" + args.model_size
    precision = args.precision
    hardware = args.hardware
    mode = f"{args.mode}_{mp.multiprocessing.cpu_count()}CORE"
    is_tcp = args.tcp

    # version = "2025_02_24"
    version = "2024_11_28"

    print("\n\n[PROGRAM] Opciones seleccionadas: ", args, "\n\n")

    model_path = f"../../models/canicas/{version}/{version}_canicas_{model_name}_{precision}_{hardware}.engine"
    # model_path = f'../../models/canicas/2024_11_28/trt/model_gn.engine'
    # video_path = '../../datasets_labeled/videos/video_muchas_canicas.mp4'
    # video_path = '../../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    # video_path = f'../../datasets_labeled/videos/contar_objetos_{objects_count}_2min.mp4'
    video_path = f"../../datasets_labeled/videos/prueba_velocidad_07_20FPS.mp4"
    output_dir = "../../inference_predictions/custom_tracker"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"prueba_velocidad_07_20FPS.mp4")

    output_hardware_stats = f"{model_name}_{precision}_{hardware}_{objects_count}_objects_{mode}"

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

    memory = {}

    print("[PROGRAM] Se ha usado el modelo ", model_path)
    # print("[PROGRAM] Total de frames: ", get_total_frames(video_path))
    print(f"[PROGRAM] Usando {objects_count} objetos, modo energia {mode}")

    stop_event = mp.Event()

    tcp_conn = mp.Event()
    t1_start = mp.Event()
    t2_start = mp.Event()

    # frame_queue = mp.Queue(maxsize=100)
    frame_queue = SharedCircularBuffer(queue_size=10, max_item_size=8)
    # detection_queue = mp.Queue(maxsize=100)
    detection_queue = SharedCircularBuffer(queue_size=100, max_item_size=4)
    # tracking_queue = mp.Queue(maxsize=100)
    tracking_queue = SharedCircularBuffer(queue_size=100, max_item_size=4)
    # times_queue = mp.Queue(maxsize=100)
    times_queue = SharedCircularBuffer(queue_size=100, max_item_size=4)

    processes = [
        mp.multiprocessing.Process(
            target=capture_frames, args=(video_path, frame_queue, stop_event, tcp_conn, is_tcp)
        ),
        mp.multiprocessing.Process(
            target=process_frames,
            args=(frame_queue, detection_queue, model_path, stop_event, t1_start),
        ),
        mp.multiprocessing.Process(
            target=tracking_frames, args=(detection_queue, tracking_queue, stop_event)
        ),
        mp.multiprocessing.Process(
            target=draw_and_write_frames,
            args=(
                tracking_queue,
                times_queue,
                output_video_path,
                CLASSES,
                memory,
                COLORS,
                stop_event,
                t2_start,
                tcp_conn,
                is_tcp,
            ),
        ),
        mp.multiprocessing.Process(target=write_to_csv, args=(times_queue, output_hardware_stats)),
        mp.multiprocessing.Process(
            target=hardware_usage,
            args=(output_hardware_stats, stop_event, t1_start, tcp_conn, is_tcp),
        ),
    ]

    for process in processes:
        process.start()

    tcp_conn.wait() if is_tcp else None
    t1_start.wait()
    t1 = cv2.getTickCount()

    t2_start.wait()
    t2 = cv2.getTickCount()

    frame_queue.close()
    frame_queue.unlink()
    detection_queue.close()
    detection_queue.unlink()
    tracking_queue.close()
    tracking_queue.unlink()
    times_queue.close()
    times_queue.unlink()

    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)

    print("[PROGRAM] Cantidad de objetos: ", objects_count)
    print(f"[PROGRAM] Total de frames procesados: {total_frames}")
    print(f"[PROGRAM] Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")


if __name__ == "__main__":
    mp.multiprocessing.set_start_method("spawn")
    print("[PROGRAM] Number of cpu : ", mp.multiprocessing.cpu_count())
    main()
