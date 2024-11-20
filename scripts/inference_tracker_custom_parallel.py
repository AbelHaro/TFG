import cv2
from ultralytics import YOLO
import os
import threading
from queue import Queue

# Variables globales para el tiempo
total_time_capturing = 0
total_time_processing = 0
total_time_writing = 0

gpu_time = {"preprocess": 0, "inference": 0, "postprocess": 0}

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def update_memory(track_id, detected_class, memory):
    if track_id not in memory:
        memory[track_id] = {'defective': detected_class.endswith('-d'), 'visible_frames': 30}
    else:
        memory[track_id]['defective'] |= detected_class.endswith('-d')
        memory[track_id]['visible_frames'] = 30  # Reset the frame counter

        if memory[track_id]['defective'] and not detected_class.endswith('-d'):
            detected_class = detected_class + '-d'

    memory[track_id]['class'] = detected_class

def capture_frames(video_path, frame_queue):
    global total_time_capturing
    cap = cv2.VideoCapture(video_path)
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

def process_frames(frame_queue, result_queue, model):
    global total_time_processing
    while True:
        frame = frame_queue.get()
        if frame is None:  # Señal de finalización
            result_queue.put(None)
            break
        t1 = cv2.getTickCount()
        results = model.track(source=frame, device=0, task='detect', tracker='bytetrack.yaml')
        t2 = cv2.getTickCount()
        total_time_processing += (t2 - t1) / cv2.getTickFrequency()

        result_queue.put((frame, results))

def draw_and_write_frames(result_queue, output_video_path, classes, memory, colors):
    global total_time_writing
    out = None
    while True:
        item = result_queue.get()
        if item is None:  # Señal de finalización
            break
        t1 = cv2.getTickCount()
        frame, results = item
        
        gpu_time['preprocess'] += results[0].speed['preprocess']
        gpu_time['inference'] += results[0].speed['inference']
        gpu_time['postprocess'] += results[0].speed['postprocess']
        
        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy().astype(float)

            for box, obj_id, cls, conf in zip(boxes, ids, detected_classes, confidences):
                xmin, ymin, xmax, ymax = box
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
    global total_time_capturing, total_time_processing, total_time_writing
    model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine'
    video_path = '../datasets_labeled/videos/video_general_defectos_3.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'video_con_tracking.mp4')

    model = YOLO(model_path)
    classes = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    colors = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }
    memory = {}

    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    threads = [
        threading.Thread(target=capture_frames, args=(video_path, frame_queue)),
        threading.Thread(target=process_frames, args=(frame_queue, result_queue, model)),
        threading.Thread(target=draw_and_write_frames, args=(result_queue, output_video_path, classes, memory, colors))
    ]

    t1 = cv2.getTickCount()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    t2 = cv2.getTickCount()
    total_time = (t2 - t1) / cv2.getTickFrequency()
    
    total_frames = get_total_frames(video_path)
    
    print(f"Total de frames procesados: {total_frames}")
    print(f"Tiempo total de procesamiento: {total_time:.3f} segundos")
    print(f"Tiempo medio por frame: {total_time / total_frames * 1000:.3f} ms, FPS: {total_frames / total_time:.3f}")
    print(f"Tiempo total capturando frames: {total_time_capturing:.3f} segundos")
    print(f"Tiempo total procesando frames: {total_time_processing:.3f} segundos")
    print(f"Tiempo total escribiendo frames: {total_time_writing:.3f} segundos")
    print(f"Tiempos que mide la función track, preprocesamiento: {gpu_time['preprocess']/1000:.3f} s, inferencia: {gpu_time['inference']/1000:.3f} s, postprocesamiento: {gpu_time['postprocess']/1000:.3f} s")
    print(f"Tiempo total que mide la función track: {(gpu_time['preprocess'] + gpu_time['inference'] + gpu_time['postprocess'])/1000:.3f} s")
    
if __name__ == "__main__":
    main()
