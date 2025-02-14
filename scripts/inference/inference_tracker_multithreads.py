import cv2
import os
import sys
from queue import Queue
import torch.multiprocessing as mp # type: ignore
from argparse import Namespace
import numpy as np
import argparse
import threading
from tcp import tcp_server, handle_send

parser = argparse.ArgumentParser()
parser.add_argument('--num_objects', default=40, type=int, choices=[0, 18, 40, 48, 60, 70, 88, 176], help='Número de objetos a contar, posibles valores: {0, 18, 40, 48, 60, 70, 88, 176}, default=40')
parser.add_argument('--model_size', default='n', type=str, choices=["n", "s", "m", "l", "x"], help='Talla del modelo {n, s, m, l, x}, default=n')
parser.add_argument('--precision', default='FP16', type=str, choices=["FP32", "FP16", "INT8"], help='Precisión del modelo {FP32, FP16, INT8}, default=FP16')
parser.add_argument('--hardware', default='GPU', type=str, choices=["GPU", "DLA0", "DLA1"], help='Hardware a usar {GPU, DLA0, DLA1}, default=GPU')
parser.add_argument('--mode', required=True, default='MAXN', type=str, choices=["MAXN", "30W", "15W", "10W"], help='Modo de energía a usar {MAXN, 30W, 15W, 10W}, default=MAXN')
parser.add_argument('--tcp', default=False, type=bool, help='Usar conexión TCP, default=False')

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

def capture_frames(video_path, frame_queue):
            
    if not os.path.exists(video_path):
        frame_queue.put(None)
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        frame_queue.put(None)
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    
    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        
        if not ret:
            break

        t2 = cv2.getTickCount()
        total_frame_time = (t2 - t1) / cv2.getTickFrequency()
        times = {
            "capture": total_frame_time
        }
        
        frame_queue.put((frame, times))
    
    cap.release()
    frame_queue.put(None)
    
def process_frames(frame_queue, detection_queue, model_path, t1_start):
    from ultralytics import YOLO # type: ignore
    times_detect_function = {}
    
    frame_number = 0
    
    model = YOLO(model_path, task='detect')
    
    model(device="cuda:0", conf=0.5, half=True, imgsz=(640, 640), augment=True)
    
    #dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    #model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
    
    t1_start.set()
    
    
    while True:
        item = frame_queue.get()
        if item is None:
            detection_queue.put(None)
            break
        
        frame, times = item
        
        t1 = cv2.getTickCount()
        
        #Preprocesa el frame
        t1_aux = cv2.getTickCount()
        preprocessed = model.predictor.preprocess([frame])
        t2_aux = cv2.getTickCount()
        times_detect_function["preprocess"] = (t2_aux - t1_aux) / cv2.getTickFrequency()
        
        #Realiza la inferencia
        t1_aux = cv2.getTickCount()
        output = model.predictor.inference(preprocessed)
        t2_aux = cv2.getTickCount()
        times_detect_function["inference"] = (t2_aux - t1_aux) / cv2.getTickFrequency()
        
        #Postprocesa los resultados
        t1_aux = cv2.getTickCount()
        results = model.predictor.postprocess(output, preprocessed, [frame])
        t2_aux = cv2.getTickCount()
        times_detect_function["postprocess"] = (t2_aux - t1_aux) / cv2.getTickFrequency()
        
        #results = model.predict(source=frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
        
        result_formatted = Namespace(
            xywh=results[0].boxes.xywh.cpu(),
            conf=results[0].boxes.conf.cpu(),
            cls=results[0].boxes.cls.cpu()
        )
        t2 = cv2.getTickCount()
        
        processing_time = (t2 - t1) / cv2.getTickFrequency()
        
        times["processing"] = processing_time
        times["detect_function"] = times_detect_function

        detection_queue.put((frame, result_formatted, times))
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"[PROGRAM - PROCESS FRAMES] Frame {frame_number} procesado", end="\r", flush=True)
    
from ultralytics.trackers.byte_tracker import BYTETracker # type: ignore
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


def tracking_frames(detection_queue, tracking_queue):

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
    
def draw_and_write_frames(tracking_queue, times_queue, output_video_path, classes, memory, colors, t2_start, finish_event):
    import threading
    import time
    
    FPS_COUNT = 0
    FPS_LABEL = 0
    out = None
    #first_time = True
    
    frame_number = 0
    
    # Función para resetear FPS cada segundo
    ##def reset_fps():
    #    nonlocal FPS_COUNT, FPS_LABEL
    #    while not stop_event.is_set():
    #        times_queue.put(("fps", FPS_COUNT))
    #        FPS_LABEL = FPS_COUNT
    #        FPS_COUNT = 0
    #        time.sleep(1)
            
            
    #client_socket, server_socker = tcp_server("0.0.0.0", 8765)
    #threading.Thread(target=handle_send, args=(client_socket, "READY"), daemon=True).start()    

    while True:
        item = tracking_queue.get()
        if item is None:
            break
        
        t1 = cv2.getTickCount()
        frame, tracked_objects, times = item
        
        #if first_time:
        #    first_time = False
        #    fps_reset_thread = threading.Thread(target=reset_fps, daemon=True)
        #    fps_reset_thread.start()

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        # Actualiza la memoria con objetos rastreados
        update_memory(tracked_objects, memory, classes)
        
        #msg_sended = False
        
        # Dibuja objetos rastreados en el frame
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
            conf = float(obj[5])
            
            if conf < 0.4:
                continue
            
            detected_class = memory[obj_id]['class']
            #if detected_class.endswith('-d') and not msg_sended:
                #threading.Thread(target=handle_send, args=(client_socket, "DETECTED_DEFECT"), daemon=True).start()
                #msg_sended = True
            color = colors.get(detected_class, (255, 255, 255))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'ID:{obj_id} {detected_class} {conf:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        #cv2.putText(frame, f'FPS: {FPS_LABEL}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        FPS_COUNT += 1
        
        t2 = cv2.getTickCount()
        writing_time = (t2 - t1) / cv2.getTickFrequency()
        times["writing"] = writing_time
        
        frame_number += 1
        
        if frame_number % 100 == 0:
            print(f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado", end="\r", flush=True)
                
        times_queue.put(("times", times))

    if out:
        out.release()
        
    times_queue.put(None)
    print("[PROGRAM - DRAW AND WRITE] None añadido a la cola de tiempos")
    
    #client_socket.close()
    #server_socker.close()
    
    t2_start.set()
    finish_event.set()

def write_to_csv(times_queue, output_file):
    from create_excel_multiprocesses import create_csv_file, add_row_to_csv, add_fps_to_csv, create_excel_from_csv
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
            
    create_excel_from_csv(times_name, fps_name, output_name=f"multiprocesses_{output_file}_2min.xlsx")
    
    print("[PROGRAM - WRITE TO CSV] None recibido, terminando proceso")
    
def hardware_usage(output_file, t1_start, finish_event):
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

    # Iniciar el proceso de tegrastats
    process = subprocess.Popen(
        ["tegrastats", "--interval", "100", "--logfile", tegra_stats_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    print("[PROGRAM - HARDWARE USAGE] Iniciando tegrastats...")
    
    finish_event.wait()

    process.terminate()
    process.wait()
    #finally:
    print("[PROGRAM - HARDWARE USAGE] Deteniendo el proceso tegrastats...")
    create_tegrastats_file(tegra_stats_output, output_excel_filename)
    print("[PROGRAM - HARDWARE USAGE] Proceso tegrastats detenido.")
    
    print("[PROGRAM - HARDWARE USAGE] Terminando proceso")

def main():
    
    objects_count = args.num_objects
    model_name = "yolo11" + args.model_size
    precision = args.precision
    hardware = args.hardware
    mode = f"{args.mode}_{mp.multiprocessing.cpu_count()}CORE"
    is_tcp = args.tcp
    
    print("\n\n[PROGRAM] Opciones seleccionadas: ", args, "\n\n")
    
    model_path = f'../../models/canicas/2024_11_28/2024_11_28_canicas_{model_name}_{precision}_{hardware}.engine'
    #model_path = f'../../models/canicas/2024_11_28/trt/model_gn.engine'
    #video_path = '../../datasets_labeled/videos/video_muchas_canicas.mp4'
    #video_path = '../../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    video_path = f'../../datasets_labeled/videos/contar_objetos_{objects_count}_2min.mp4'
    output_dir = '../../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f'multithread_{model_name}_{precision}_{hardware}_{objects_count}_objects_{mode}.mp4')
    
    output_hardware_stats = f"{model_name}_{precision}_{hardware}_{objects_count}_objects_{mode}"
    

    CLASSES = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    COLORS = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }

    memory = {}
    
    print("[PROGRAM] Se ha usado el modelo ", model_path)
    print("[PROGRAM] Total de frames: ", get_total_frames(video_path))
    print(f"[PROGRAM] Usando {objects_count} objetos, modo energia {mode}")
    

    t1_start = mp.Event()
    t2_start = mp.Event()
    finish_event = mp.Event()
    
    frame_queue = Queue(maxsize=10)
    detection_queue = Queue(maxsize=100)
    tracking_queue = Queue(maxsize=10)
    times_queue = Queue(maxsize=10)

    threads = [
            threading.Thread(target=capture_frames, args=(video_path, frame_queue,)),
            threading.Thread(target=process_frames, args=(frame_queue, detection_queue, model_path, t1_start)),
            threading.Thread(target=tracking_frames, args=(detection_queue, tracking_queue)),
            threading.Thread(target=draw_and_write_frames, args=(tracking_queue, times_queue, output_video_path, CLASSES, memory, COLORS, t2_start, finish_event)),
            threading.Thread(target=write_to_csv, args=(times_queue,output_hardware_stats)),
            threading.Thread(target=hardware_usage, args=(output_hardware_stats, t1_start, finish_event)),
        ]

    for thread in threads:
        thread.start()
    
    t1_start.wait()
    t1 = cv2.getTickCount()
    
    t2_start.wait()
    t2 = cv2.getTickCount()
    
    for thread in threads:
        thread.join()
    
    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)
    
    print("[PROGRAM] Cantidad de objetos: ", objects_count)
    print(f"[PROGRAM] Total de frames procesados: {total_frames}")
    print(f"[PROGRAM] Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")
        
    
if __name__ == '__main__':
    main()
