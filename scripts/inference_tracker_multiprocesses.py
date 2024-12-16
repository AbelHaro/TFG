import cv2
from ultralytics import YOLO
import os
import torch.multiprocessing as mp
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace
import numpy as np
from create_excel import create_excel

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
        
        is_defective = detected_class.endswith('-d')
        if track_id in memory:
            entry = memory[track_id]
            entry['defective'] |= is_defective
            entry['visible_frames'] = FRAME_AGE
            if entry['defective'] and not is_defective:
                detected_class += '-d'
            entry['class'] = detected_class
        else:
            memory[track_id] = {'defective': is_defective, 'visible_frames': FRAME_AGE, 'class': detected_class}
            
    for track_id in list(memory):
        memory[track_id]['visible_frames'] -= 1
        if memory[track_id]['visible_frames'] <= 0:
            del memory[track_id]

def capture_frames(video_path, frame_queue, stop_event, times_queue):
    capture_times = []
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    while cap.isOpened():
        #t1 = cv2.getTickCount()
        ret, frame = cap.read()
        #t2 = cv2.getTickCount()
        
        #capture_time = (t2 - t1) / cv2.getTickFrequency()
        #capture_times.append(capture_time)

        if not ret:
            break
        frame_queue.put(frame)
    
    cap.release()
    frame_queue.put(None)
    
    
    #times_queue.put(("capture", capture_times))
    
    while not stop_event.is_set():
        pass

    

def process_frames(frame_queue, detection_queue, model_path, stop_event, times_queue):
    times_detect_function = {"preprocess": 0, "inference": 0, "postprocess": 0}
    processing_times = []
    model = YOLO(model_path, task='detect')
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            detection_queue.put(None)
            break

        #t1 = cv2.getTickCount()
        results = model.predict(source=frame, device=0, conf=0.6, imgsz=(640, 640), half=True, augment=True, task='detect', show_labels=False, show_conf=False, )
        result_formatted = Namespace(
            xywh=results[0].boxes.xywh.cpu(),
            conf=results[0].boxes.conf.cpu(),
            cls=results[0].boxes.cls.cpu()
        )
        #t2 = cv2.getTickCount()

        detection_queue.put((frame, result_formatted))
        
        #processing_time = (t2 - t1) / cv2.getTickFrequency()
        #processing_times.append(processing_time)
        #times_detect_function["preprocess"] += results[0].speed["preprocess"]
        #times_detect_function["inference"] += results[0].speed["inference"]
        #times_detect_function["postprocess"] += results[0].speed["postprocess"]
        
        
    #times_queue.put(("processing", processing_times))
    #times_queue.put(("times_detect_function", times_detect_function))
    
    
    
    while not stop_event.is_set():
        pass
    
    print("Proceso de procesamiento terminado, he marcado el evento de parada")

class TrackerWrapper:
    global FRAME_AGE
    def __init__(self, frame_rate=20):
        self.args = Namespace(
            tracker_type='bytetrack',
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=FRAME_AGE,
            match_thresh=0.8,
            fuse_score=True
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
            detection_data.cls.numpy().astype(int)
        )
        return self.tracker.update(detections, frame)


def tracking_frames(detection_queue, tracking_queue, stop_event, times_queue):
    tracking_times = []
    objects_counts = []
    tracker_wrapper = TrackerWrapper(frame_rate=20)
    
    while True:
        item = detection_queue.get()
        if item is None:
            tracking_queue.put(None)
            break

        #t1 = cv2.getTickCount()
        frame, result = item

        outputs = tracker_wrapper.track(result, frame)

        #t2 = cv2.getTickCount()
        tracking_queue.put((frame, outputs))
        
        #tracking_time = (t2 - t1) / cv2.getTickFrequency()
        #tracking_times.append(tracking_time)

        #objects_counts.append(len(outputs))
        
        
    times_queue.put(("tracking", tracking_times))
    #times_queue.put(("objects_counts", objects_counts))
    
    while not stop_event.is_set():
        pass
    
    os._exit(0)
    
def draw_and_write_frames(tracking_queue, output_video_path, classes, memory, colors, stop_event, times_queue):
    writting_times = []
    out = None
    
    while True:
        item = tracking_queue.get()
        if item is None:
            break
        #t1 = cv2.getTickCount()
        frame, tracked_objects = item

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        update_memory(tracked_objects, memory, classes)
        
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
            conf = float(obj[5])
            
            if conf < 0.4:
                continue
            
            detected_class = memory[obj_id]['class']
            color = colors.get(detected_class, (255, 255, 255))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'ID:{obj_id} {detected_class} {conf:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(frame)
        
        #t2 = cv2.getTickCount()
        #writting_time = (t2 - t1) / cv2.getTickFrequency()
        #writting_times.append(writting_time)

    if out:
        out.release()
    
    print("Proceso de escritura terminado, voy a escribir en la cola mis tiempos")
    times_queue.put(("writting", writting_times))
    print("Proceso de escritura terminado, he escrito en la cola mis tiempos")
    stop_event.set()
    print("Proceso de escritura terminado, he marcado el evento de parada")
    
    os._exit(0)

def main():
    model_path = '../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16.engine'
    video_path = '../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'multiprocesos.mp4')
    

    classes = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    colors = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }

    memory = {}
    
    print("Total de frames: ", get_total_frames(video_path))
    
    capture_times = []
    processing_times = []
    tracking_times = []
    writting_times = []
    objects_counts = []
    times_detect_function = []
    
    stop_event = mp.Event()
    
    frame_queue = mp.multiprocessing.Queue(maxsize=10)
    detection_queue = mp.multiprocessing.Queue(maxsize=10)
    tracking_queue = mp.multiprocessing.Queue(maxsize=10)
    
    times_queue = mp.multiprocessing.Queue(maxsize=50)

    t1 = cv2.getTickCount()
    processes = [
            mp.multiprocessing.Process(target=capture_frames, args=(video_path, frame_queue, stop_event, times_queue)),
            mp.multiprocessing.Process(target=process_frames, args=(frame_queue, detection_queue, model_path, stop_event, times_queue)),
            mp.multiprocessing.Process(target=tracking_frames, args=(detection_queue, tracking_queue, stop_event, times_queue)),
            mp.multiprocessing.Process(target=draw_and_write_frames, args=(tracking_queue, output_video_path, classes, memory, colors, stop_event, times_queue))
        ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()
        print("Proceso ", process, " joined")

    t2 = cv2.getTickCount()
    
    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)
    
    print("Se ha usado el modelo ", model_path)
    
    print(f"Total de frames procesados: {total_frames}")
    print(f"Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")
    
    while not times_queue.qsize() == 0:
        print("El tamaño de la cola de tiempos es: ", times_queue.qsize())
        label, data = times_queue.get()
        print("Se ha sacado de la cola el siguiente dato: ", label)
        if label == "capture":
            capture_times = data
        if label == "processing":
            processing_times = data
        if label == "tracking":
            tracking_times = data
        if label == "writting":
            writting_times = data
        if label == "objects_counts":
            objects_counts = data
        if label == "times_detect_function":
            times_detect_function = data
        else:
            print("No se ha encontrado el tiempo de detección")
        
    print("Se ha pasado a la lista de tiempos el siguiente dato: ")
    
    total_times = {
        "Captura": sum(capture_times),
        "Procesamiento": sum(processing_times),
        "Tracking": sum(tracking_times),
        "Escritura": sum(writting_times)
    }

    print("Tiempos totales: ")
    
    max_task = max(total_times, key=total_times.get)
    print(f"Mayor tiempo de ejecución: {max_task}, con un tiempo de {total_times[max_task]:.3f}s")
    
    print("Creando archivo excel...")
    
    times = {
        "capture": capture_times,
        "processing": processing_times,
        "tracking": tracking_times,
        "writting": writting_times,
        "objects_count": objects_counts,
        "frames_per_second": []
    }

    create_excel(times, total_frames, file="times_multiprocesses.csv")
        
    
if __name__ == '__main__':
    mp.multiprocessing.set_start_method('spawn')   
    print("Number of cpu : ", mp.multiprocessing.cpu_count())
    main()