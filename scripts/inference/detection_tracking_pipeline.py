from abc import ABC, abstractmethod
import cv2
import os
from argparse import Namespace
from classes.tracker_wrapper import TrackerWrapper
from lib.tcp import handle_send, tcp_server
import logging
from typing import Union, Optional
import torch.multiprocessing as mp
from classes.shared_circular_buffer import SharedCircularBuffer
from lib.constants import TIMING_FIELDS
import time

DEFAULT_SAHI_CONFIG = {
    "slice_width": 640,
    "slice_height": 640,
    "overlap_pixels": 200,
    "iou_threshold": 0.4,
    "conf_threshold": 0.5,
    "overlap_threshold": 0.8,
    "batch_size": 4,
}


class DetectionTrackingPipeline(ABC):
    """Clase base para pipelines de detección y tracking.
    
    Esta clase implementa la funcionalidad común para todos los tipos de pipelines
    de detección y tracking, permitiendo diferentes estrategias de paralelización.
    """

    # Configuración de logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Configuración por defecto de SAHI

    # Clases y colores para visualización desde config
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

    def get_total_frames(self, video_path: str) -> int:
        """Obtiene el número total de frames en un video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error al abrir el archivo de video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def update_memory(self, tracked_objects, memory, classes) -> None:
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

                # Si alcanza el umbral, lo marcamos como defectuoso permanente
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

        # Limpieza de memoria (objetos no vistos en FRAME_AGE frames)
        for track_id in list(memory):
            memory[track_id]["visible_frames"] -= 1
            if memory[track_id]["visible_frames"] <= 0:
                del memory[track_id]

    def capture_frames( 
            self,
            video_path: str,
            frame_queue: Union[mp.Queue, SharedCircularBuffer],
            t1_start: mp.Event,
            stop_event: mp.Event,
            tcp_event: mp.Event,
            is_tcp: bool,
            mp_stop_event: Optional[mp.Event] = None,
            mh_num: int = 1,
            is_process: bool = False,
            max_frames: Optional[int] = None,
            ):
            logging.debug(f"[PROGRAM - CAPTURE FRAMES] Iniciando captura de frames")

            if not os.path.exists(video_path):
                for _ in range(mh_num):
                    frame_queue.put(None)
                raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

            cap = cv2.VideoCapture(video_path)
            # cap = cv2.VideoCapture('/dev/video0')
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # cap.set(cv2.CAP_PROP_FPS, 30)
            
            first_time = True

            if not cap.isOpened():
                frame_queue.put(None)
                raise IOError(f"Error al abrir el archivo de video: {video_path}")

            tcp_event.wait() if is_tcp else None

            frame_count = 0

            import time
            t1_start.wait()
            frame_time = 1 / max_frames if max_frames else None

            while cap.isOpened() and not stop_event.is_set():
                loop_start_time = time.time()
                
                if first_time:
                    t1 = cv2.getTickCount()
                    first_time = False
                    
                ret, frame = cap.read()

                if not ret:
                    logging.debug(
                        f"[PROGRAM - CAPTURE FRAMES] No se pudo leer el frame, añadiendo None a la cola"
                    )
                    logging.debug(f"[PROGRAM - CAPTURE FRAMES] Se han procesado {frame_count} frames")
                    break

                    
                t2 = cv2.getTickCount()
                total_frame_time = (t2 - t1) / cv2.getTickFrequency()
                t1 = cv2.getTickCount()
                
                times = {TIMING_FIELDS["CAPTURE"]: total_frame_time}
                # print(f"[DEBUG] Poniendo frame a la cola", frame.shape)
                try:
                    frame_queue.put((frame, times, frame_count)) if not max_frames else frame_queue.put_nowait((frame, times, frame_count))
                
                except Exception as e:
                    pass
                    
                
                elapsed_time = time.time() - loop_start_time
                if max_frames and elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                    
                frame_count += 1
                               
            cap.release()
            logging.debug(f"[PROGRAM - CAPTURE FRAMES] Captura de frames terminada")
            for _ in range(mh_num):
                frame_queue.put(None)

            mp_stop_event.wait() if mp_stop_event else None
            os._exit(0) if is_process else None

    def process_frames(
        self,
        frame_queue: Union[mp.Queue, SharedCircularBuffer],
        detection_queue: Union[mp.Queue, SharedCircularBuffer],
        model_path: str,
        t1_start: mp.Event,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        from ultralytics import YOLO  # type: ignore

        times_detect_function = {}

        model = YOLO(model_path, task="detect")

        model(conf=0.5, half=True, imgsz=(640, 640), augment=True)

        # dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')

        t1_start.set()

        while True:
            item = frame_queue.get()
            if item is None:
                detection_queue.put(None)
                break

            frame, times, frame_count = item

            t1 = cv2.getTickCount()

            # Preprocesa el frame
            t1_aux = cv2.getTickCount()
            preprocessed = model.predictor.preprocess([frame])
            t2_aux = cv2.getTickCount()
            times_detect_function[TIMING_FIELDS["PREPROCESS"]] = (
                t2_aux - t1_aux
            ) / cv2.getTickFrequency()

            # Realiza la inferencia
            t1_aux = cv2.getTickCount()
            output = model.predictor.inference(preprocessed)
            t2_aux = cv2.getTickCount()
            times_detect_function[TIMING_FIELDS["INFERENCE"]] = (
                t2_aux - t1_aux
            ) / cv2.getTickFrequency()

            # Postprocesa los resultados
            t1_aux = cv2.getTickCount()
            results = model.predictor.postprocess(output, preprocessed, [frame])
            t2_aux = cv2.getTickCount()
            times_detect_function[TIMING_FIELDS["POSTPROCESS"]] = (
                t2_aux - t1_aux
            ) / cv2.getTickFrequency()

            # results = model.predict(source=frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')

            result_formatted = Namespace(
                xywh=results[0].boxes.xywh.cpu(),
                conf=results[0].boxes.conf.cpu(),
                cls=results[0].boxes.cls.cpu(),
            )
            t2 = cv2.getTickCount()

            processing_time = (t2 - t1) / cv2.getTickFrequency()

            times[TIMING_FIELDS["PROCESSING"]] = processing_time
            times[TIMING_FIELDS["DETECT_FUNCTION"]] = times_detect_function

            detection_queue.put((frame, result_formatted, times, frame_count))

        mp_stop_event.wait() if mp_stop_event else None
        logging.debug(f"[PROGRAM - PROCESS FRAMES] Procesamiento de frames terminado")

        os._exit(0) if is_process else None

    def process_frames_sahi(
        self,
        frame_queue: Union[mp.Queue, SharedCircularBuffer],
        detection_queue: Union[mp.Queue, SharedCircularBuffer],
        model_path: str,
        t1_start: mp.Event,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        from ultralytics import YOLO  # type: ignore
        from lib.sahi import (
            split_image_with_overlap,
            apply_nms,
            process_detection_results,
            apply_overlapping,
            apply_nms_custom,
        )

        try:
            import torch
            import numpy as np
            from typing import List, Tuple
        except ImportError as e:
            raise ImportError(f"Error importing required packages: {e}")

        times_detect_function = {}
        model = YOLO(model_path, task="detect")

        # Configuración de SAHI desde las constantes de clase
        new_width = DEFAULT_SAHI_CONFIG["slice_width"]
        new_height = DEFAULT_SAHI_CONFIG["slice_height"]
        overlap_pixels = DEFAULT_SAHI_CONFIG["overlap_pixels"]
        batch_size = DEFAULT_SAHI_CONFIG["batch_size"]

        # Warm up del modelo con batch
        dummy_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        dummy_batch = [dummy_frame] * batch_size
        model.predict(dummy_batch, conf=0.5, half=True, augment=True, batch=batch_size)
        t1_start.set()

        while True:
            item = frame_queue.get()
            # print(f"[DEBUG] Item recibido: {item}")
            if item is None:
                detection_queue.put(None)
                break

            frame, times, frame_count = item
            t1 = cv2.getTickCount()

            t1_aux = cv2.getTickCount()

            sub_images, horizontal_splits, vertical_splits = split_image_with_overlap(
                frame, new_width, new_height, overlap_pixels
            )

            t2_aux = cv2.getTickCount()
            # logging.debug(f"[PROGRAM - PROCESS FRAMES] Tiempo de división de imagen: {((t2_aux - t1_aux) / cv2.getTickFrequency()) * 1000:.2f} ms")

            t1_aux = cv2.getTickCount()
            results = model.predict(
                sub_images, conf=0.5, half=True, augment=True, batch=4, verbose=False
            )
            t2_aux = cv2.getTickCount()
            # logging.debug(f"[PROGRAM - PROCESS FRAMES] Tiempo de inferencia: {((t2_aux - t1_aux) / cv2.getTickFrequency()) * 1000:.2f} ms")

            t1_aux = cv2.getTickCount()

            transformed_results = process_detection_results(
                results,
                horizontal_splits,
                vertical_splits,
                new_width,
                new_height,
                overlap_pixels,
                frame.shape[1],
                frame.shape[0],
            )

            # Se aplica NMS a los resultados
            nms_results = apply_nms_custom(
                transformed_results, iou_threshold=0.4, conf_threshold=0.5
            )

            # Se aplica NMS con solapamiento a los resultados
            final_results = apply_overlapping(nms_results, overlap_threshold=0.8)

            # Convertir final_results a array de numpy para procesamiento vectorizado
            if final_results:
                # Convertir la lista de tuplas a un array numpy
                results_array = np.array(final_results)

                # Extraer componentes [cls, conf, xmin, ymin, xmax, ymax]
                classes = results_array[:, 0]
                confidences = results_array[:, 1]
                boxes = results_array[:, 2:6]  # [xmin, ymin, xmax, ymax]

                # Calcular xywh de forma vectorizada
                widths = boxes[:, 2] - boxes[:, 0]  # xmax - xmin
                heights = boxes[:, 3] - boxes[:, 1]  # ymax - ymin
                x_centers = (boxes[:, 0] + boxes[:, 2]) / 2  # (xmin + xmax) / 2
                y_centers = (boxes[:, 1] + boxes[:, 3]) / 2  # (ymin + ymax) / 2

                # Apilar para crear el array xywh
                xywh = np.stack([x_centers, y_centers, widths, heights], axis=1)

                # Convertir a tensores PyTorch
                xywh_tensor = torch.from_numpy(xywh).float()
                confidences_tensor = torch.from_numpy(confidences).float()
                classes_tensor = torch.from_numpy(classes).float()
            else:
                # Si no hay detecciones, crear tensores vacíos
                xywh_tensor = torch.empty((0, 4), dtype=torch.float32)
                confidences_tensor = torch.empty(0, dtype=torch.float32)
                classes_tensor = torch.empty(0, dtype=torch.float32)

            # Crear el objeto Namespace con los resultados formateados
            result_formatted = Namespace(
                xywh=xywh_tensor,  # Tensor de 2D: [N, 4]
                conf=confidences_tensor,  # Tensor de 2D: [N, 1]
                cls=classes_tensor,  # Tensor de 2D: [N, 1]
            )

            # Mostrar los resultados formateados
            # print("Resultados formateados:", result_formatted)
            t2_aux = cv2.getTickCount()
            # logging.debug(f"[PROGRAM - PROCESS FRAMES] Tiempo de postprocesamiento: {((t2_aux - t1_aux) / cv2.getTickFrequency()) * 1000:.2f} ms")

            times_detect_function["preprocess"] = results[0].speed["preprocess"]
            times_detect_function["inference"] = results[0].speed["inference"]
            times_detect_function["postprocess"] = results[0].speed["postprocess"]

            # Medir el tiempo de procesamiento
            t2 = cv2.getTickCount()
            processing_time = (t2 - t1) / cv2.getTickFrequency()

            # Actualizar el diccionario de tiempos
            times["processing"] = processing_time
            times["detect_function"] = times_detect_function

            detection_queue.put((frame, result_formatted, times, frame_count))

        mp_stop_event.wait() if mp_stop_event else None
        logging.debug(f"[PROGRAM - PROCESS FRAMES] Procesamiento de frames terminado")

        os._exit(0) if is_process else None

    def tracking_frames(
        self,
        detection_queue: Union[mp.Queue, SharedCircularBuffer],
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        tracker_wrapper = TrackerWrapper(frame_rate=30)

        while True:
            item = detection_queue.get()

            if item is None:
                tracking_queue.put(None)
                break

            t1 = cv2.getTickCount()
            frame, result, times, _ = item

            outputs = tracker_wrapper.track(result, frame)

            t2 = cv2.getTickCount()

            tracking_time = (t2 - t1) / cv2.getTickFrequency()

            times[TIMING_FIELDS["TRACKING"]] = tracking_time
            times[TIMING_FIELDS["OBJECTS_COUNT"]] = len(outputs)

            tracking_queue.put((frame, outputs, times))

        mp_stop_event.wait() if mp_stop_event else None
        logging.debug(f"[PROGRAM - TRACKING FRAMES] Tracking de frames terminado")

        os._exit(0) if is_process else None

    def tracking_frames_multihardware(
        self,
        detection_queue_GPU: Union[mp.Queue, SharedCircularBuffer],
        detection_queue_DLA0: Union[mp.Queue, SharedCircularBuffer],
        detection_queue_DLA1: Union[mp.Queue, SharedCircularBuffer],
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        tracker_wrapper = TrackerWrapper(frame_rate=30)
        stop_gpu = False
        stop_dla0 = False
        stop_dla1 = False
        item_gpu = None
        item_dla0 = None
        item_dla1 = None

        while True:
            # Obtener elementos de las colas si aún no han sido detenidas
            if not stop_gpu and item_gpu is None:
                item_gpu = detection_queue_GPU.get()
                if item_gpu is None:
                    stop_gpu = True

            if not stop_dla0 and item_dla0 is None:
                item_dla0 = detection_queue_DLA0.get()
                if item_dla0 is None:
                    stop_dla0 = True

            if not stop_dla1 and item_dla1 is None:
                item_dla1 = detection_queue_DLA1.get()
                if item_dla1 is None:
                    stop_dla1 = True

            # Verificar si todas las colas están vacías
            if stop_gpu and stop_dla0 and stop_dla1:
                tracking_queue.put(None)
                mp_stop_event.wait() if mp_stop_event else None
                logging.debug(f"[PROGRAM - TRACKING FRAMES] Tracking de frames terminado")
                os._exit(0) if is_process else None
                break

            t1 = cv2.getTickCount()

            # Obtener los números de frame de los ítems
            _, _, _, frame_number_gpu = (
                item_gpu if item_gpu is not None else (None, None, None, None)
            )
            _, _, _, frame_number_dla0 = (
                item_dla0 if item_dla0 is not None else (None, None, None, None)
            )
            _, _, _, frame_number_dla1 = (
                item_dla1 if item_dla1 is not None else (None, None, None, None)
            )

            # Determinar qué ítem procesar (prioridad por número de frame más bajo)
            if frame_number_gpu is not None and (
                (frame_number_dla0 is None or frame_number_gpu < frame_number_dla0)
                and (frame_number_dla1 is None or frame_number_gpu < frame_number_dla1)
            ):
                frame, result, times, _ = item_gpu
                item_gpu = None
            elif frame_number_dla0 is not None and (
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

    def draw_and_write_frames(
        self,
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        times_queue: Union[mp.Queue, SharedCircularBuffer],
        output_video_path: str,
        classes: dict,
        memory: dict,
        colors: dict,
        stop_event: mp.Event,
        tcp_conn: mp.Event,
        is_tcp: bool,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        # Crear un ThreadPoolExecutor para reutilizar los threads
        thread_pool = ThreadPoolExecutor(max_workers=8)  # Ajustar según el número de núcleos

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
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

            # Actualiza la memoria con objetos rastreados
            self.update_memory(tracked_objects, memory, classes)

            msg_sended = False

            # Función para dibujar un objeto en el frame
            def draw_object(obj):
                nonlocal frame, memory, colors, msg_sended, is_tcp, client_socket
                
                xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
                conf = float(obj[5])

                if conf < 0.4:
                    return

                detected_class = memory[obj_id]["class"]
                if detected_class.endswith("-d") and not msg_sended and is_tcp:
                    threading.Thread(
                        target=handle_send,
                        args=(client_socket, "DETECTED_DEFECT"),
                        daemon=True,
                    ).start()
                    msg_sended = True
                color = colors.get(detected_class, (255, 255, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                text = f"ID:{obj_id} {detected_class} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Usar thread pool para dibujar objetos en paralelo
            futures = [thread_pool.submit(draw_object, obj) for obj in tracked_objects]
            
            # Esperar a que todas las tareas terminen
            for future in futures:
                future.result()

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out.write(frame)
            FPS_COUNT += 1

            #cv2.imshow("Frame", frame)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break

            t2 = cv2.getTickCount()
            writing_time = (t2 - t1) / cv2.getTickFrequency()
            times["writing"] = writing_time

            frame_number += 1

            # print(f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado")

            if frame_number % 30 == 0:
                print(
                    f"[PROGRAM - DRAW AND WRITE] Frame {frame_number} procesado",
                    end="\r",
                    flush=True,
                )

            times_queue.put(("times", times))

        if out:
            out.release()
            # Cerrar el thread pool al finalizar
            thread_pool.shutdown()

        times_queue.put(None)
        stop_event.set()
        mp_stop_event.wait() if mp_stop_event else None
        logging.debug(f"[PROGRAM - DRAW AND WRITE] Escritura de frames terminada")

        if is_tcp:
            client_socket.close()
            server_socker.close()

        os._exit(0) if is_process else None

    def write_to_csv(
        self,
        times_queue: Union[mp.Queue, SharedCircularBuffer],
        output_file: str,
        parallel_mode: str,
        t1_start: mp.Event,
        stop_event: mp.Event,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        from lib.create_excel import (
            create_csv_file,
            add_row_to_csv,
            add_fps_to_csv,
            create_excel_from_csv,
        )
        from lib.hardware_stats_usage import create_tegrastats_file

        times_name = "times_aux.csv"
        fps_name = "fps_aux.csv"

        times_excel_file = create_csv_file(parallel_mode, file_name=times_name)
        fps_excel_file = create_csv_file(parallel_mode, file_name=fps_name)

        frame_count = 0
        
        t1_start.wait()
        t1 = cv2.getTickCount()

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

        tegra_stats_output = f"/TFG/excels/{parallel_mode}/aux_files/hardware_usage.txt"
        hardware_usage_name = f"/TFG/excels/{parallel_mode}/aux_files/hardware_usage_aux.csv"
        hardware_usage_file = "hardware_usage_aux.csv"

        stop_event.wait()
        t2 = cv2.getTickCount()
        total_time = (t2 - t1) / cv2.getTickFrequency()
        
        mp_stop_event.set() if mp_stop_event else None


        print(f"[PROGRAM - WRITE TO CSV] Total time de write_to_csv: {total_time:.2f} s")
        create_tegrastats_file(tegra_stats_output, hardware_usage_name, total_time)

        create_excel_from_csv(
            times_name,
            fps_name,
            hardware_usage_file,
            parallel_mode,
            output_name=f"{parallel_mode}_{output_file}_2min.xlsx",
        )

        logging.debug(f"[PROGRAM - WRITE TO CSV] Escritura de tiempos terminada")
        
        print(f"[PROGRAM - WRITE TO CSV] Se han procesado {frame_count} frames")

        os._exit(0) if is_process else None

    def hardware_usage(
        self,
        parallel_mode: str,
        stop_event: mp.Event,
        t1_start: mp.Event,
        tcp_event: mp.Event,
        is_tcp: bool = False,
        is_process: bool = False,
    ):
        import subprocess
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        tegra_stats_output = f"/TFG/excels/{parallel_mode}/aux_files/hardware_usage.txt"
        output_excel_filename = f"/TFG/excels/{parallel_mode}/aux_files/hardware_usage.csv"

        os.makedirs(os.path.dirname(tegra_stats_output), exist_ok=True)
        os.makedirs(os.path.dirname(output_excel_filename), exist_ok=True)

        # Espera inicial para sincronizar con el evento
        t1_start.wait()
        tcp_event.wait() if is_tcp else None

        # Iniciar el proceso de tegrastats
        process = subprocess.Popen(
            ["tegrastats", "--interval", "10", "--logfile", tegra_stats_output],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logging.debug(f"[PROGRAM - HARDWARE USAGE] Iniciando tegrastats...")

        stop_event.wait()
        process.terminate()
        process.wait()

        logging.debug(f"[PROGRAM - HARDWARE USAGE] Proceso tegrastats detenido.")

        os._exit(0) if is_process else None

    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_video_path: str,
        output_times: str,
        parallel_mode: str,
        is_tcp: bool = False,
        sahi: bool = False,
        max_fps: int = None,
        mh_num: int = 1,
        is_process: bool = True,
    ):
        """Inicializa el pipeline con la configuración común."""
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.is_tcp = is_tcp
        self.sahi = sahi
        self.max_fps = max_fps
        self.mh_num = mh_num
        self.is_process = is_process

        # Eventos de control comunes
        self.tcp_event = mp.Event()
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()
        self.mp_stop_event = mp.Event() if is_process else None

        # Memoria compartida
        self.memory = {}

    def _initialize_queues(self):
        """Método abstracto para inicializar las colas según el tipo de pipeline."""
        pass

    def _create_processes(self):
        """Método abstracto para crear los procesos/hilos del pipeline."""
        pass

    def _cleanup(self):
        """Método abstracto para limpieza de recursos."""
        pass

    @abstractmethod
    def run(self):
        """Ejecuta el pipeline. Debe ser implementado por las clases derivadas."""
        pass
