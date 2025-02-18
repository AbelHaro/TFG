import cv2
import os
import sys
import torch.multiprocessing as mp # type: ignore
from argparse import Namespace
import numpy as np
import argparse
from tcp import tcp_server, handle_send
from multiprocessing import shared_memory, Lock, Value


parser = argparse.ArgumentParser()
parser.add_argument('--num_objects', default=40, type=int, choices=[0, 18, 40, 48, 60, 70, 88, 176], help='Número de objetos a contar, posibles valores: {0, 18, 40, 48, 60, 70, 88, 176}, default=40')
parser.add_argument('--model_size', default='n', type=str, choices=["n", "s", "m", "l", "x"], help='Talla del modelo {n, s, m, l, x}, default=n')
parser.add_argument('--precision', default='FP16', type=str, choices=["FP32", "FP16", "INT8"], help='Precisión del modelo {FP32, FP16, INT8}, default=FP16')
parser.add_argument('--hardware', default='GPU', type=str, choices=["GPU", "DLA0", "DLA1"], help='Hardware a usar {GPU, DLA0, DLA1}, default=GPU')
parser.add_argument('--mode', required=True, default='MAXN', type=str, choices=["MAXN", "30W", "15W", "10W"], help='Modo de energía a usar {MAXN, 30W, 15W, 10W}, default=MAXN')
parser.add_argument('--tcp', default=False, type=bool, help='Usar conexión TCP, default=False')

args = parser.parse_args()

FRAME_AGE = 15

import pickle
import multiprocessing
from multiprocessing import shared_memory, Lock, Value

class SharedCircularBuffer:
    def __init__(self, queue_size=10, max_item_size=1, name=None):
        """
        Inicializa un buffer circular en memoria compartida.
        
        :param queue_size: Número máximo de elementos en la cola.
        :param max_item_size: Tamaño máximo en MB de cada elemento.
        :param name: Nombre de la memoria compartida (None para crear una nueva).
        """
        self.queue_size = queue_size

        if max_item_size not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError("El tamaño máximo del item debe ser 1, 2, 4, 8, 16, 32, 64, 128, 256 o 512 MB.")
        
        self.max_item_size = max_item_size * 1024 * 1024
        self.total_size = queue_size * self.max_item_size

        if name:
            self.shm = shared_memory.SharedMemory(name=name)
        else:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)

        self.head = Value("i", 0)  # Índice de lectura
        self.tail = Value("i", 0)  # Índice de escritura
        self.count = Value("i", 0)  # Número de elementos en la cola
        self.lock = Lock()

    def enqueue(self, item):
        """Agrega un item a la cola en memoria compartida."""
        # Verificar si el item es un array y aplanarlo si es necesario
        if isinstance(item, np.ndarray):
            reshaped_item = item.reshape(-1)
            item_data = {"data": reshaped_item, "shape": item.shape}
        else:
            item_data = {"data": item, "shape": None}  # No es un array

        data_bytes = pickle.dumps(item_data)
        #print(f"[DEBUG] Empaquetando item: {item_data}")
        #print(f"[DEBUG] Tamaño del item: {len(data_bytes)} bytes.")

        if len(data_bytes) > self.max_item_size:
            raise ValueError("El item es demasiado grande para la cola.")

        with self.lock:
            if self.count.value == self.queue_size:
                #print("[WARNING] Cola llena, sobrescribiendo el elemento más antiguo.")
                self.head.value = (self.head.value + 1) % self.queue_size  # Avanza el head

            pos = (self.tail.value % self.queue_size) * self.max_item_size
            #print(f"[DEBUG] Escribiendo en posición {pos} de la memoria compartida.")

            # Escritura corregida
            self.shm.buf[pos : pos + len(data_bytes)] = memoryview(data_bytes)

            self.tail.value = (self.tail.value + 1) % self.queue_size
            self.count.value = min(self.count.value + 1, self.queue_size)

    def dequeue(self):
        """Extrae un item de la cola en memoria compartida."""
        with self.lock:
            if self.count.value == 0:
                return None  # Cola vacía

            pos = (self.head.value % self.queue_size) * self.max_item_size
            data_bytes = bytes(self.shm.buf[pos:pos+self.max_item_size])  # Leer memoria
            
            self.head.value = (self.head.value + 1) % self.queue_size
            self.count.value -= 1

        # Deserializar
        item_data = pickle.loads(data_bytes)
        #print(f"[DEBUG] Desempaquetando item: {item_data}")

        # Si tiene una forma almacenada, significa que era un array y se debe reconstruir
        if item_data["shape"] is not None:
            return np.array(item_data["data"]).reshape(item_data["shape"])
        
        return item_data["data"]  # Si no era un array, devolver el dato normal


    def is_empty(self):
        """Retorna True si la cola está vacía."""
        with self.lock:
            return self.count.value == 0

    def is_full(self):
        """Retorna True si la cola está llena."""
        with self.lock:
            return self.count.value == self.queue_size

    def close(self):
        """Cierra la memoria compartida."""
        self.shm.close()

    def unlink(self):
        """Libera la memoria compartida (solo debe llamarse una vez)."""
        self.shm.unlink()
        

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

def capture_frames(video_path, frame_queue, stop_event, tcp_conn, is_tcp):
    
        
    print(f"[DEBUG] Iniciando captura de frames")
        
    if not os.path.exists(video_path):
        frame_queue.enqueue(None)
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        frame_queue.enqueue(None)
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    tcp_conn.wait() if is_tcp else None
    
    while cap.isOpened() and not stop_event.is_set():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        
        if not ret:
            print("[PROGRAM - CAPTURE FRAMES] No se pudo leer el frame, añadiendo None a la cola")
            break

        t2 = cv2.getTickCount()
        total_frame_time = (t2 - t1) / cv2.getTickFrequency()
        times = {
            "capture": total_frame_time
        }
        print(f"[DEBUG] Poniendo frame a la cola", frame.shape)
        frame_queue.enqueue((frame, times))
    
    cap.release()
    print("[PROGRAM - CAPTURE FRAMES] Video terminado, añadiendo None a la cola")
    frame_queue.enqueue(None)
    
    while not stop_event.is_set():
        pass

def process_frames(frame_queue, detection_queue, model_path, stop_event, t1_start):
    from ultralytics import YOLO # type: ignore
    times_detect_function = {}
    
    model = YOLO(model_path, task='detect')
    
    model(device="cuda:0", conf=0.5, half=True, imgsz=(640, 640), augment=True)
    
    #dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    #model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
    
    t1_start.set()
    
    
    while True:
        item = frame_queue.dequeue()
        print(f"[DEBUG] Item recibido: {item}")
        if item is None:
            detection_queue.enqueue(None)
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

        detection_queue.enqueue((frame, result_formatted, times))
    
    while not stop_event.is_set():
        pass

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


def tracking_frames(detection_queue, tracking_queue, stop_event):

    tracker_wrapper = TrackerWrapper(frame_rate=20)
    
    while True:
        item = detection_queue.dequeue()
        if item is None:
            tracking_queue.enqueue(None)
            break

        t1 = cv2.getTickCount()
        frame, result, times = item

        outputs = tracker_wrapper.track(result, frame)

        t2 = cv2.getTickCount()
        
        tracking_time = (t2 - t1) / cv2.getTickFrequency()
        
        times["tracking"] = tracking_time
        times["objects_count"] = len(outputs)
    
        tracking_queue.enqueue((frame, outputs, times))
        
    while not stop_event.is_set():
        pass
    
    os._exit(0)
    
def draw_and_write_frames(tracking_queue, times_queue, output_video_path, classes, memory, colors, stop_event, t2_start, tcp_conn, is_tcp):
    import threading
    import time
    
    FPS_COUNT = 0
    FPS_LABEL = 0
    out = None
    first_time = True
    
    frame_number = 0
    
    # Función para resetear FPS cada segundo
    def reset_fps():
        nonlocal FPS_COUNT, FPS_LABEL
        while not stop_event.is_set():
            times_queue.enqueue(("fps", FPS_COUNT))
            FPS_LABEL = FPS_COUNT
            FPS_COUNT = 0
            time.sleep(1)
            
    if is_tcp:
        client_socket, server_socker = tcp_server("0.0.0.0", 8765)
        threading.Thread(target=handle_send, args=(client_socket, "READY"), daemon=True).start()
    
    tcp_conn.set() if is_tcp else None
    

    while True:
        item = tracking_queue.dequeue()
        if item is None:
            break
        
        t1 = cv2.getTickCount()
        frame, tracked_objects, times = item
        
        if first_time:
            first_time = False
            fps_reset_thread = threading.Thread(target=reset_fps, daemon=True)
            fps_reset_thread.start()

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        # Actualiza la memoria con objetos rastreados
        update_memory(tracked_objects, memory, classes)
        
        msg_sended = False
        
        # Dibuja objetos rastreados en el frame
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
            conf = float(obj[5])
            
            if conf < 0.4:
                continue
            
            detected_class = memory[obj_id]['class']
            if detected_class.endswith('-d') and not msg_sended and is_tcp:
                threading.Thread(target=handle_send, args=(client_socket, "DETECTED_DEFECT"), daemon=True).start()
                msg_sended = True
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
        
        if frame_number % 20 == 0:
            print(f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado", end="\r", flush=True)
                
        times_queue.enqueue(("times", times))

    if out:
        out.release()
        
    times_queue.enqueue(None)
    print("[PROGRAM - DRAW AND WRITE] None añadido a la cola de tiempos")
    
    
    if is_tcp:
        client_socket.close()
        server_socker.close()
    
    t2_start.set()
    stop_event.set()
    os._exit(0)


def write_to_csv(times_queue, output_file):
    from create_excel_multiprocesses import create_csv_file, add_row_to_csv, add_fps_to_csv, create_excel_from_csv
    import os
    
    frame_count = 0
    
    times_name = "times_multiprocesses.csv"
    fps_name = "fps_multiprocesses.csv"
    
    times_excel_file = create_csv_file(file_name=times_name)
    fps_excel_file = create_csv_file(file_name=fps_name)
    
    while True:
        item = times_queue.dequeue()
        
        if item is None:
            break

        label, data = item
        
        if label == "times":
            add_row_to_csv(times_excel_file, frame_count, data)
        elif label == "fps":
            add_fps_to_csv(fps_excel_file, frame_count, data)
            
    create_excel_from_csv(times_name, fps_name, output_name=f"multiprocesses_{output_file}_2min.xlsx")
    
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
        ["tegrastats", "--interval", "100", "--logfile", tegra_stats_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    print("[PROGRAM - HARDWARE USAGE] Iniciando tegrastats...")

    stop_event.wait()
    process.terminate()
    process.wait()
    #finally:
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
    
    print("\n\n[PROGRAM] Opciones seleccionadas: ", args, "\n\n")
        
    model_path = f'../../models/canicas/2024_11_28/2024_11_28_canicas_{model_name}_{precision}_{hardware}.engine'
    #model_path = f'../../models/canicas/2024_11_28/trt/model_gn.engine'
    #video_path = '../../datasets_labeled/videos/video_muchas_canicas.mp4'
    #video_path = '../../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    video_path = f'../../datasets_labeled/videos/contar_objetos_{objects_count}_2min.mp4'
    output_dir = '../../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f'multiprocesos_{model_name}_{precision}_{hardware}_{objects_count}_objects_{mode}.mp4')
    
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
    
    stop_event = mp.Event()
    
    tcp_conn = mp.Event()
    t1_start = mp.Event()
    t2_start = mp.Event()
    
    frame_queue = SharedCircularBuffer(queue_size=100, max_item_size=128)
    detection_queue = SharedCircularBuffer(queue_size=100, max_item_size=128)
    tracking_queue = SharedCircularBuffer(queue_size=10, max_item_size=128)
    times_queue = SharedCircularBuffer(queue_size=10, max_item_size=32)

    processes = [
            mp.multiprocessing.Process(target=capture_frames, args=(video_path, frame_queue, stop_event, tcp_conn, is_tcp)),
            mp.multiprocessing.Process(target=process_frames, args=(frame_queue, detection_queue, model_path, stop_event, t1_start)),
            mp.multiprocessing.Process(target=tracking_frames, args=(detection_queue, tracking_queue, stop_event)),
            mp.multiprocessing.Process(target=draw_and_write_frames, args=(tracking_queue, times_queue, output_video_path, CLASSES, memory, COLORS, stop_event, t2_start, tcp_conn, is_tcp)),
            mp.multiprocessing.Process(target=write_to_csv, args=(times_queue,output_hardware_stats)),
            mp.multiprocessing.Process(target=hardware_usage, args=(output_hardware_stats, stop_event, t1_start, tcp_conn, is_tcp)),
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
        
    
if __name__ == '__main__':
    mp.multiprocessing.set_start_method('spawn')   
    print("[PROGRAM] Number of cpu : ", mp.multiprocessing.cpu_count())
    main()
