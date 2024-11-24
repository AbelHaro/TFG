import cv2
from ultralytics import YOLO
import os
import threading
from queue import Queue
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace

# Variables globales para el tiempo
total_time_capturing = 0
total_time_processing = 0
total_time_tracking = 0
total_time_writing = 0
times_detect_function = {"preprocess": 0, "inference": 0, "postprocess": 0}


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def update_memory(track_id, detected_class, memory):
    if track_id not in memory:
        memory[track_id] = {'defective': detected_class.endswith('-d'), 'visible_frames': 30}
    else:
        memory[track_id]['defective'] |= detected_class.endswith('-d')
        memory[track_id]['visible_frames'] = 30  # Reset counter

        if memory[track_id]['defective'] and not detected_class.endswith('-d'):
            detected_class = detected_class + '-d'

    memory[track_id]['class'] = detected_class

def capture_frames(video_path, frame_queue):
    global total_time_capturing
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
    cap.release()
    frame_queue.put(None)  # Señal de finalización

def process_frames(frame_queue, detection_queue, model):
    global total_time_processing, times_detect_function
    while True:
        frame = frame_queue.get()
        if frame is None:  # Señal de finalización
            detection_queue.put(None)
            break
        t1 = cv2.getTickCount()
        results = model.predict(source=frame, device=0, task='detect')
        t2 = cv2.getTickCount()
        total_time_processing += (t2 - t1) / cv2.getTickFrequency()
        detection_queue.put((frame, results[0]))
        times_detect_function["preprocess"] += results[0].speed["preprocess"]
        times_detect_function["inference"] += results[0].speed["inference"]
        times_detect_function["postprocess"] += results[0].speed["postprocess"]

def tracking_frames(detection_queue, tracking_queue):
    global total_time_tracking
    args = Namespace(
        tracker_type='bytetrack',
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True
    )
    tracker = BYTETracker(args, frame_rate=30)

    class Detections:
        def __init__(self, boxes, confidences, class_ids):
            self.conf = confidences
            self.xywh = boxes
            self.cls = class_ids
            
    while True:
        item = detection_queue.get()
        if item is None:
            tracking_queue.put(None) # Señal de finalización
            break

        t1 = cv2.getTickCount()
        frame, result = item

        detections = Detections(
            result.boxes.xywh.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int)
        )
        
        outputs = tracker.update(detections, frame)
        t2 = cv2.getTickCount()
        total_time_tracking += (t2 - t1) / cv2.getTickFrequency()
        tracking_queue.put((frame, outputs))

def draw_and_write_frames(tracking_queue, output_video_path, classes, memory, colors):
    global total_time_writing
    out = None
    while True:
        item = tracking_queue.get()
        if item is None:
            break
        t1 = cv2.getTickCount()
        frame, tracked_objects = item

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])  # Asignamos las primeras 5 posiciones a enteros
            conf = float(obj[5])  # Convertimos el valor de conf a float
            cls = int(obj[6])  # Asignamos cls como entero
         
            detected_class = classes[cls]

            update_memory(obj_id, detected_class, memory)
            detected_class = memory[obj_id]['class']
            color = colors.get(detected_class, (255, 255, 255))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'ID:{obj_id} {detected_class} {conf:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for track_id in list(memory):
            memory[track_id]['visible_frames'] -= 1
            if memory[track_id]['visible_frames'] <= 0:
                del memory[track_id]

        out.write(frame)
        t2 = cv2.getTickCount()
        total_time_writing += (t2 - t1) / cv2.getTickFrequency()

    if out:
        out.release()

def main():
    model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine'
    video_path = '../datasets_labeled/videos/video_general_defectos_3.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'video_con_tracking.mp4')

    classes = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    colors = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }
    memory = {}
    
    model = YOLO(model_path)

    frame_queue = Queue(maxsize=10)
    detection_queue = Queue(maxsize=10)
    tracking_queue = Queue(maxsize=10)

    threads = [
        threading.Thread(target=capture_frames, args=(video_path, frame_queue)),
        threading.Thread(target=process_frames, args=(frame_queue, detection_queue, model)),
        threading.Thread(target=tracking_frames, args=(detection_queue, tracking_queue)),
        threading.Thread(target=draw_and_write_frames, args=(tracking_queue, output_video_path, classes, memory, colors))
    ]

    t1 = cv2.getTickCount()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    t2 = cv2.getTickCount()

    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)
    
    total_time_detect_function = (times_detect_function["preprocess"] + times_detect_function["inference"] + times_detect_function["postprocess"]) / 1000

    print(f"Total de frames procesados: {total_frames}")
    print(f"Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")
    print(f"Captura: {total_time_capturing:.3f}s, Procesamiento: {total_time_processing:.3f}s")
    print(f"Tracking: {total_time_tracking:.3f}s, Escritura: {total_time_writing:.3f}s")
    print(f"Tiempo total función detect: {total_time_detect_function:.3f}s")
    print(f"Tiempo de preprocesamiento: {times_detect_function['preprocess']/1000:.3f}s")
    print(f"Tiempo de inferencia: {times_detect_function['inference']/1000:.3f}s")
    print(f"Tiempo de postprocesamiento: {times_detect_function['postprocess']/1000:.3f}s")

if __name__ == '__main__':
    main()
