
import cv2
import os
import sys
import torch.multiprocessing as mp # type: ignore
from argparse import Namespace
import numpy as np

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

def capture_frames(video_path, frame_queue, stop_event):
    frame_number = 0
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        t2 = cv2.getTickCount()
        
        capture_time = (t2 - t1) / cv2.getTickFrequency()

        if not ret:
            break
        
        times = {
            "capture": capture_time
        }
        
        frame_queue.put((frame, times, frame_number))
    
    cap.release()
    frame_queue.put(None)
    frame_queue.put(None)
    
    #time.sleep(1)
    
    while not stop_event.is_set():
        pass

    

def process_frames(frame_queue, detection_queue, model_path, stop_event, t1_start):
    from ultralytics import YOLO # type: ignore
    times_detect_function = {}
    
    model = YOLO(model_path, task='detect')
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
    
    t1_start.set()
    
    while True:
        item = frame_queue.get()
        if item is None:
            detection_queue.put(None)
            break
        
        frame, times, frame_number = item
        
        t1 = cv2.getTickCount()
        results = model.predict(source=frame, device=0, conf=0.6, imgsz=(640, 640), half=True, augment=True, task='detect', show_labels=False, show_conf=False, )
        result_formatted = Namespace(
            xywh=results[0].boxes.xywh.cpu(),
            conf=results[0].boxes.conf.cpu(),
            cls=results[0].boxes.cls.cpu()
        )
        t2 = cv2.getTickCount()
        
        
        processing_time = (t2 - t1) / cv2.getTickFrequency()
        times_detect_function["preprocess"] = results[0].speed["preprocess"]
        times_detect_function["inference"] = results[0].speed["inference"]
        times_detect_function["postprocess"] = results[0].speed["postprocess"]
        
        times["processing"] = processing_time
        times["detect_function"] = times_detect_function

        detection_queue.put((frame, result_formatted, times, frame_number))
        
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


def tracking_frames(detection_queue_DLA0, detection_queue_DLA1, tracking_queue, stop_event):

    tracker_wrapper = TrackerWrapper(frame_rate=20)
    stop_dla0 = False
    stop_dla1 = False
    item_dla0 = None
    item_dla1 = None
    
    while True:
        if not stop_dla0 and item_dla0 is None:
            item_dla0 = detection_queue_DLA0.get()
            if item_dla0 is None:
                stop_dla0 = True
        
        if not stop_dla1 and item_dla1 is None:
            item_dla1 = detection_queue_DLA1.get()
            if item_dla1 is None:
                stop_dla1 = True

        # Verificar si todas las colas están vacías
        if stop_dla0 and stop_dla1:
            tracking_queue.put(None)
            break

        t1 = cv2.getTickCount()

        # Obtener los números de frame de los ítems
        _, _, _, frame_number_dla0 = item_dla0 if item_dla0 is not None else (None, None, None, None)
        _, _, _, frame_number_dla1 = item_dla1 if item_dla1 is not None else (None, None, None, None)

        # Determinar qué ítem procesar (prioridad por número de frame más bajo)
        if frame_number_dla0 is not None and (
            (frame_number_dla1 is None or frame_number_dla0 < frame_number_dla1)
        ):
            frame, result, times, _ = item_dla0
            item_dla0 = None
        elif frame_number_dla1 is not None:
            frame, result, times, _ = item_dla1
            item_dla1 = None
        else:
            continue  # No hay elementos disponibles, continuar el bucle
        
        # Realizar el tracking
        outputs = tracker_wrapper.track(result, frame)
        
        t2 = cv2.getTickCount()
        tracking_time = (t2 - t1) / cv2.getTickFrequency()
        
        # Actualizar tiempos y contar objetos
        times["tracking"] = tracking_time
        times["objects_count"] = len(outputs)
        
        # Añadir el resultado al tracking_queue
        tracking_queue.put((frame, outputs, times))
    
    # Esperar hasta que se establezca el evento de parada
    while not stop_event.is_set():
        pass

    os._exit(0)
    
def draw_and_write_frames(tracking_queue, times_queue, output_video_path, classes, memory, colors, stop_event, t2_start):
    import threading
    import time
    
    FPS_COUNT = 0
    FPS_LABEL = 0
    frames_per_second_record = []
    out = None
    first_time = True
    
    # Función para resetear FPS cada segundo
    def reset_fps():
        """
        Calcula y escribe los FPS en el archivo CSV cada segundo.
        """
        nonlocal FPS_COUNT, FPS_LABEL
        while not stop_event.is_set():
            frames_per_second_record.append(FPS_COUNT)
            # Escribe el valor del FPS en el archivo directamente
            times_queue.put(("fps", FPS_COUNT))
            FPS_LABEL = FPS_COUNT
            FPS_COUNT = 0
            time.sleep(1)

    while True:
        item = tracking_queue.get()
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
        
        # Dibuja objetos rastreados en el frame
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
            
        #cv2.putText(frame, f'FPS: {FPS_LABEL}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        FPS_COUNT += 1
        
        t2 = cv2.getTickCount()
        writing_time = (t2 - t1) / cv2.getTickFrequency()
        times["writing"] = writing_time
                
        times_queue.put(("times", times))

    if out:
        out.release()
        
    times_queue.put(None)
    print("[PROGRAM - DRAW AND WRITE] None añadido a la cola de tiempos")    

    t2_start.set()
    stop_event.set()
    os._exit(0)


def write_to_csv(times_queue, model_size, objects_count):
    from create_excel_multiprocesses import create_csv_file, add_row_to_csv, add_fps_to_csv, create_excel_from_csv
    import os
    
    times_name = "times_multiprocesses.csv"
    fps_name = "fps_multiprocesses.csv"
    
    times_excel_file = create_csv_file(file_name=times_name)
    fps_excel_file = create_csv_file(file_name=fps_name)
    
    frame_count = 0
    
    while True:
        item = times_queue.get()
        
        if item is None:
            break

        label, data = item
        
        if label == "times":
            add_row_to_csv(times_excel_file, frame_count, data)
            frame_count += 1
        elif label == "fps":
            add_fps_to_csv(fps_excel_file, frame_count, data)
            
    create_excel_from_csv(times_name, fps_name, output_name=f"multiprocesses-{model_size}-contar_objetos_{objects_count}_2min.xlsx")
    
    print("[PROGRAM - WRITE TO CSV] None recibido, terminando proceso")
        
    os._exit(0)
    




def hardware_usage(output_file, stop_event, t1_start):
    import subprocess
    from datetime import datetime
    from hardware_stats_usage import create_tegrastats_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    tegra_stats_output = f"/TFG/excels/tegrastats_outputs/{output_file}_{timestamp}.txt"
    output_excel_filename = f"/TFG/excels/hardware_stats_usage/{output_file}"
    
    os.makedirs(os.path.dirname(tegra_stats_output), exist_ok=True)
    os.makedirs(os.path.dirname(output_excel_filename), exist_ok=True)
    
    # Espera inicial para sincronizar con el evento
    t1_start.wait()

    # Iniciar el proceso de tegrastats
    process = subprocess.Popen(
        ["tegrastats", "--interval", "100", "--logfile", tegra_stats_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    print("[PROGRAM - HARDWARE USAGE] Iniciando tegrastats...")
    #try:
        # Leer la salida en tiempo real y escribir en el archivo
    #    with open(tegra_stats_output, "a") as f:
    #        while not stop_event.is_set():
    #            output = process.stdout.readline()
    #            if output:
    #                f.write(output)
    #            else:
    #                break
        # Detener el proceso cuando el evento de parada se activa
    
    stop_event.wait()
    process.terminate()
    process.wait()
    #finally:
    print("[PROGRAM - HARDWARE USAGE] Deteniendo el proceso tegrastats...")
    # Procesar el archivo de salida generado por tegrastats
    create_tegrastats_file(tegra_stats_output, output_excel_filename)
    print("[PROGRAM - HARDWARE USAGE] Proceso tegrastats detenido.")
    
    print("[PROGRAM - HARDWARE USAGE] Terminando proceso")
    os._exit(0)




def main():

    if len(sys.argv) < 2:
        print("Usage: python3 inference_tracker_multiprocesses.py <objects_count>")
        exit()
        
    objects_count = str(sys.argv[1])
    
    model_name = "yolo11n"
    precision = "FP16"
    hardware_DLA0 = "DLA0"
    hardware_DLA1 = "DLA1"
    mode = f"MAXN_{mp.multiprocessing.cpu_count()}CORE"
    
    model_path_DLA0 = f'../../models/canicas/2024_11_28/2024_11_28_canicas_{model_name}_{precision}_{hardware_DLA0}.engine'
    model_path_DLA1 = f'../../models/canicas/2024_11_28/2024_11_28_canicas_{model_name}_{precision}_{hardware_DLA1}.engine'
    #model_path = f'../../models/canicas/2024_11_28/trt/model_gn.engine'
    #video_path = '../../datasets_labeled/videos/video_muchas_canicas.mp4'
    #video_path = '../../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    video_path = f'../../datasets_labeled/videos/contar_objetos_{objects_count}_2min.mp4'
    output_dir = '../../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'multiprocesos_tensorRT_manual_export_DLA0.mp4')
    
    output_hardware_stats = f"{model_name}_{precision}_{hardware_DLA0}-{hardware_DLA1}_{objects_count}_objects_{mode}.csv"
    

    CLASSES = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    COLORS = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }

    memory = {}
    
    print("[PROGRAM] Se ha usado el modelo ", model_path_DLA0)
    print("[PROGRAM] Se ha usado el modelo ", model_path_DLA1)
    print("[PROGRAM] Total de frames: ", get_total_frames(video_path))
    print(f"[PROGRAM] Usando {objects_count} objetos, modo energia {mode}")
    
    stop_event = mp.Event()
    t1_start = mp.Event()
    t2_start = mp.Event()
    
    frame_queue = mp.Queue(maxsize=10)
    detection_queue_DLA0 = mp.Queue(maxsize=10)
    detection_queue_DLA1 = mp.Queue(maxsize=10)
    tracking_queue = mp.Queue(maxsize=10)
    times_queue = mp.Queue(maxsize=10)

    processes = [
            mp.multiprocessing.Process(target=capture_frames, args=(video_path, frame_queue, stop_event)),
            mp.multiprocessing.Process(target=process_frames, args=(frame_queue, detection_queue_DLA0, model_path_DLA0, stop_event, t1_start)),
            mp.multiprocessing.Process(target=process_frames, args=(frame_queue, detection_queue_DLA1, model_path_DLA1, stop_event, t1_start)),
            mp.multiprocessing.Process(target=tracking_frames, args=(detection_queue_DLA0, detection_queue_DLA1, tracking_queue, stop_event)),
            mp.multiprocessing.Process(target=draw_and_write_frames, args=(tracking_queue, times_queue, output_video_path, CLASSES, memory, COLORS, stop_event, t2_start)),
            mp.multiprocessing.Process(target=write_to_csv, args=(times_queue,model_name, objects_count)),
            mp.multiprocessing.Process(target=hardware_usage, args=(output_hardware_stats, stop_event, t1_start)),
        ]

    for process in processes:
        process.start()
        
    t1_start.wait()
    t1 = cv2.getTickCount()
    
    t2_start.wait()
    t2 = cv2.getTickCount()
    
    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)
    
    print("[PROGRAM] Cantidad de objetos: ", objects_count)
    print(f"[PROGRAM] Total de frames procesados: {total_frames}")
    print(f"[PROGRAM] Tiempo total: {total_time:.3f}s, FPS: {total_frames / total_time:.3f}")
        
    
if __name__ == '__main__':
    mp.multiprocessing.set_start_method('spawn')   
    print("[PROGRAM] Number of cpu : ", mp.multiprocessing.cpu_count())
    main()
