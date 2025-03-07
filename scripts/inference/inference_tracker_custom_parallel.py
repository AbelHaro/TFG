import cv2
from ultralytics import YOLO
import os
import threading
from queue import Queue
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace
import time
import numpy as np
from create_excel import create_excel

FRAME_AGE = 15

# Variables globales para el tiempo
total_time_capturing = 0
total_time_processing = 0
total_time_tracking = 0
total_time_writting = 0
times_detect_function = {"preprocess": 0, "inference": 0, "postprocess": 0}

frames_per_second_counter = 0
lock = threading.Lock()
frame_count_finish = False

capture_times = []
processing_times = []
tracking_times = []
writting_times = []
objects_counts = []
frames_per_second_record = []
preprocess_times = []
inference_times = []
postprocess_times = []


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

        is_defective = detected_class.endswith('-d')
        if track_id in memory:
            entry = memory[track_id]
            entry['defective'] |= is_defective
            entry['visible_frames'] = FRAME_AGE
            if entry['defective'] and not is_defective:
                detected_class += '-d'
            entry['class'] = detected_class
        else:
            memory[track_id] = {
                'defective': is_defective,
                'visible_frames': FRAME_AGE,
                'class': detected_class,
            }

    for track_id in list(memory):
        memory[track_id]['visible_frames'] -= 1
        if memory[track_id]['visible_frames'] <= 0:
            del memory[track_id]


def capture_frames(video_path, frame_queue):
    global total_time_capturing, capture_times
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")

    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        t2 = cv2.getTickCount()
        total_time_capturing += (t2 - t1) / cv2.getTickFrequency()
        if not ret:
            break
        frame_queue.put(frame)
        capture_times.append((t2 - t1) / cv2.getTickFrequency())

    cap.release()
    frame_queue.put(None)


def process_frames(frame_queue, detection_queue, model):
    global total_time_processing, times_detect_function, processing_times
    while True:
        frame = frame_queue.get()
        if frame is None:
            detection_queue.put(None)
            break
        t1 = cv2.getTickCount()
        results = model.predict(
            source=frame,
            device=0,
            conf=0.2,
            imgsz=(640, 640),
            half=True,
            augment=True,
            task='detect',
        )
        t2 = cv2.getTickCount()
        total_time_processing += (t2 - t1) / cv2.getTickFrequency()
        detection_queue.put((frame, results[0]))
        times_detect_function["preprocess"] += results[0].speed["preprocess"]
        preprocess_times.append(results[0].speed["preprocess"])
        times_detect_function["inference"] += results[0].speed["inference"]
        inference_times.append(results[0].speed["inference"])
        times_detect_function["postprocess"] += results[0].speed["postprocess"]
        postprocess_times.append(results[0].speed["postprocess"])
        processing_times.append((t2 - t1) / cv2.getTickFrequency())


class TrackerWrapper:
    global FRAME_AGE

    def __init__(self, frame_rate=20):
        # Definir los argumentos para BYTETracker
        self.args = Namespace(
            tracker_type='bytetrack',
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=FRAME_AGE,
            match_thresh=0.8,
            fuse_score=True,
        )
        # Crear instancia del tracker
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    class Detections:
        def __init__(self, boxes, confidences, class_ids):
            self.conf = confidences
            self.xywh = boxes
            self.cls = class_ids

    def track(self, detection_data, frame):
        # Convertir detecciones para el tracker
        detections = self.Detections(
            detection_data.boxes.xywh.cpu().numpy(),
            detection_data.boxes.conf.cpu().numpy(),
            detection_data.boxes.cls.cpu().numpy().astype(int),
        )
        # Actualizar el tracker con las detecciones actuales
        return self.tracker.update(detections, frame)


def tracking_frames(detection_queue, tracking_queue):
    global total_time_tracking, tracking_times, objects_counts

    # Instanciar el tracker
    tracker_wrapper = TrackerWrapper(frame_rate=20)

    while True:
        item = detection_queue.get()
        if item is None:  # Señal de finalización
            tracking_queue.put(None)
            break

        t1 = cv2.getTickCount()
        frame, result = item

        # Usar el método de la clase para realizar el tracking
        outputs = tracker_wrapper.track(result, frame)

        t2 = cv2.getTickCount()
        total_time_tracking += (t2 - t1) / cv2.getTickFrequency()
        tracking_queue.put((frame, outputs))
        tracking_times.append((t2 - t1) / cv2.getTickFrequency())
        objects_counts.append(len(outputs))


def draw_and_write_frames(tracking_queue, output_video_path, classes, memory, colors):
    global total_time_writting, writting_times, frames_per_second_counter, lock, frame_count_finish

    time_updating_memory = 0
    time_drawing = 0

    frame_number = 0
    out = None
    while True:
        item = tracking_queue.get()
        if item is None:
            frame_count_finish = True
            break
        t1 = cv2.getTickCount()

        frame, tracked_objects = item

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        update_memory(tracked_objects, memory, classes)

        t_aux = cv2.getTickCount()

        time_updating_memory += (t_aux - t1) / cv2.getTickFrequency()

        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(
                int, obj[:5]
            )  # Asignamos las primeras 5 posiciones a enteros
            conf = float(obj[5])  # Convertimos el valor de conf a float

            if conf < 0.4:
                continue

            detected_class = memory[obj_id]['class']
            color = colors.get(detected_class, (255, 255, 255))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'ID:{obj_id} {detected_class} {conf:.2f}'
            cv2.putText(
                frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        cv2.putText(frame, str(frame_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_number += 1

        with lock:
            frames_per_second_counter += 1

        out.write(frame)

        # Mostrar el video en pantalla
        # cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t2 = cv2.getTickCount()
        total_time_writting += (t2 - t1) / cv2.getTickFrequency()
        writting_times.append((t2 - t1) / cv2.getTickFrequency())

        time_drawing += (t2 - t_aux) / cv2.getTickFrequency()

    if out:
        out.release()

    print("El tiempo de actualización de memoria fue de ", time_updating_memory)
    print("El tiempo de dibujado fue de ", time_drawing)

    cv2.destroyAllWindows()


def frames_per_second():
    global frames_per_second_counter, frames_per_second_record, frame_count_finish

    while not frame_count_finish:
        with lock:
            frames_per_second_record.append(frames_per_second_counter)
            frames_per_second_counter = 0

        time.sleep(1)


def main():
    model_path = '../../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16_GPU.engine'
    video_path = '../../datasets_labeled/videos/contar_objetos_variable_2min.mp4'
    output_dir = '../../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'hilos_video_con_tracking.mp4')

    classes = {
        0: 'negra',
        1: 'blanca',
        2: 'verde',
        3: 'azul',
        4: 'negra-d',
        5: 'blanca-d',
        6: 'verde-d',
        7: 'azul-d',
    }
    colors = {
        'negra': (0, 0, 255),
        'blanca': (0, 255, 0),
        'verde': (255, 0, 0),
        'azul': (255, 255, 0),
        'negra-d': (0, 165, 255),
        'blanca-d': (255, 165, 0),
        'verde-d': (255, 105, 180),
        'azul-d': (255, 0, 255),
    }

    memory = {}

    model = YOLO(model_path, task='detect')

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    print("Haciendo una predicción para cargar el modelo en memoria...")

    # Realizar una predicción para cargar el modelo en memoria
    model.predict(
        source=dummy_frame,
        device=0,
        conf=0.2,
        imgsz=(640, 640),
        half=True,
        augment=True,
        task='detect',
    )

    frame_queue = Queue(maxsize=10)
    detection_queue = Queue(maxsize=10)
    tracking_queue = Queue(maxsize=10)

    threads = [
        threading.Thread(target=capture_frames, args=(video_path, frame_queue)),
        threading.Thread(target=process_frames, args=(frame_queue, detection_queue, model)),
        threading.Thread(target=tracking_frames, args=(detection_queue, tracking_queue)),
        threading.Thread(
            target=draw_and_write_frames,
            args=(tracking_queue, output_video_path, classes, memory, colors),
        ),
        threading.Thread(target=frames_per_second),
    ]

    t1 = cv2.getTickCount()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    t2 = cv2.getTickCount()

    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)

    total_time_detect_function = (
        times_detect_function["preprocess"]
        + times_detect_function["inference"]
        + times_detect_function["postprocess"]
    ) / 1000

    print("Se ha usado el modelo ", model_path)

    print(f"Total de frames procesados: {total_frames}")
    print(f"Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")
    print(f"Captura: {total_time_capturing:.3f}s, Procesamiento: {total_time_processing:.3f}s")
    print(f"Tracking: {total_time_tracking:.3f}s, Escritura: {total_time_writting:.3f}s")
    print("-" * 50)
    print(f"Tiempo total función detect: {total_time_detect_function:.3f}s")
    print(f"Tiempo de preprocesamiento: {times_detect_function['preprocess']/1000:.3f}s")
    print(f"Tiempo de inferencia: {times_detect_function['inference']/1000:.3f}s")
    print(f"Tiempo de postprocesamiento: {times_detect_function['postprocess']/1000:.3f}s")

    total_times = {
        "Captura": total_time_capturing,
        "Procesamiento": total_time_processing,
        "Tracking": total_time_tracking,
        "Escritura": total_time_writting,
    }

    max_task = max(total_times, key=total_times.get)
    print(f"El mayor tiempo fue {total_times[max_task]:.3f}s en la tarea de {max_task}.")

    times = {
        "capture": capture_times,
        "processing": processing_times,
        "tracking": tracking_times,
        "writting": writting_times,
        "objects_count": objects_counts,
        "frames_per_second": frames_per_second_record,
        "preprocess": preprocess_times,
        "inference": inference_times,
        "postprocess": postprocess_times,
    }

    create_excel(times, len(capture_times), file="paralelo_hilos.csv")


if __name__ == '__main__':
    main()
