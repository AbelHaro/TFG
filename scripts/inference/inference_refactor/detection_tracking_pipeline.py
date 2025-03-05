from abc import ABC, abstractmethod
import cv2
import os
import Namespace
from tracker_wrapper import TrackerWrapper
from tcp import handle_send, tcp_server

class DetectionTrackingPipeline(ABC):
    
    CLASSES = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    
    COLORS = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }
    
    FRAME_AGE = 20
    
    def get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error al abrir el archivo de video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
    def update_memory(self, tracked_objects, memory, classes):
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
    
    def capture_frames(self, video_path, frame_queue, stop_event, tcp_conn, is_tcp):
        print(f"[DEBUG] Iniciando captura de frames")
            
        if not os.path.exists(video_path):
            frame_queue.put(None)
            raise FileNotFoundError(f"El archivo de video no existe: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            frame_queue.put(None)
            raise IOError(f"Error al abrir el archivo de video: {video_path}")
        
        tcp_conn.wait() if is_tcp else None
        
        frame_count = 0
        
        while cap.isOpened() and not stop_event.is_set():
            t1 = cv2.getTickCount()
            ret, frame = cap.read()
            
            if not ret:
                print("[PROGRAM - CAPTURE FRAMES] No se pudo leer el frame, añadiendo None a la cola")
                print("[PROGRAM - CAPTURE FRAMES - DEBUG] Se han procesado", frame_count, "frames")
                break

            t2 = cv2.getTickCount()
            total_frame_time = (t2 - t1) / cv2.getTickFrequency()
            times = {
                "capture": total_frame_time
            }
            #print(f"[DEBUG] Poniendo frame a la cola", frame.shape)
            frame_queue.put((frame, times))
            frame_count += 1
        
        cap.release()
        print("[PROGRAM - CAPTURE FRAMES] Video terminado, añadiendo None a la cola")
        frame_queue.put(None)
        
    def process_frames(self, frame_queue, detection_queue, model_path, t1_start):
        from ultralytics import YOLO # type: ignore
        times_detect_function = {}
        
        model = YOLO(model_path, task='detect')
        
        model(device="cuda:0", conf=0.5, half=True, imgsz=(640, 640), augment=True)
        
        #dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        #model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
        
        t1_start.set()
        
        
        while True:
            item = frame_queue.get()
            #print(f"[DEBUG] Item recibido: {item}")
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
        
    def tracking_frames(self, detection_queue, tracking_queue):
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
        
    def draw_and_write_frames(self, tracking_queue, times_queue, output_video_path, classes, memory, colors, stop_event, tcp_conn, is_tcp):
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
                times_queue.put(("fps", FPS_COUNT))
                FPS_LABEL = FPS_COUNT
                FPS_COUNT = 0
                time.sleep(1)
                
        if is_tcp:
            client_socket, server_socker = tcp_server("0.0.0.0", 8765)
            threading.Thread(target=handle_send, args=(client_socket, "READY"), daemon=True).start()
        
        tcp_conn.set() if is_tcp else None
        

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
            self.update_memory(tracked_objects, memory, classes)
            
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
                
            cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
            FPS_COUNT += 1
            
            t2 = cv2.getTickCount()
            writing_time = (t2 - t1) / cv2.getTickFrequency()
            times["writing"] = writing_time
            
            frame_number += 1
            
            #print(f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado")
            
            if frame_number % 20 == 0:
                print(f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado", end="\r", flush=True)
                    
            times_queue.put(("times", times))

        if out:
            out.release()
            
        times_queue.put(None)
        print("[PROGRAM - DRAW AND WRITE] None añadido a la cola de tiempos")
        
        
        if is_tcp:
            client_socket.close()
            server_socker.close()

    def write_to_csv(self, times_queue, output_file):
        from create_excel_multiprocesses import create_csv_file, add_row_to_csv, add_fps_to_csv, create_excel_from_csv
        
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
        
    def hardware_usage(self, output_file, stop_event, t1_start, tcp_conn, is_tcp):
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
        
    @abstractmethod
    def run(self):
        pass
    
