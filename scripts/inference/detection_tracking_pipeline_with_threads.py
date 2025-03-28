from detection_tracking_pipeline import DetectionTrackingPipeline
from queue import Queue
import torch.multiprocessing as mp  # type: ignore
import threading
import cv2


class DetectionTrackingPipelineWithThreads(DetectionTrackingPipeline):

    def __init__(
        self,
        video_path,
        model_path,
        output_video_path,
        output_times,
        parallel_mode,
        is_tcp=False,
        sahi=False,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.is_tcp = is_tcp
        self.sahi = sahi

        # Colas específicas para hilos
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=10)
        self.tracking_queue = Queue(maxsize=10)
        self.times_queue = Queue(maxsize=10)

        # Memoria compartida
        self.memory = {}

        # Eventos de control
        self.tcp_event = mp.Event()
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()

        # Parámetros adicionales
        self.is_process = False
        self.mh_num = 1

    def run(self):
        # Definir los hilos
        threads = [
            threading.Thread(
                target=self.capture_frames,
                args=(
                    self.video_path,
                    self.frame_queue,
                    self.t1_start,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    None,  # mp_stop_event no usado en hilos
                    self.mh_num,
                    self.is_process,
                ),
            ),
            threading.Thread(
                target=self.process_frames_sahi if self.sahi else self.process_frames,
                args=(
                    self.frame_queue,
                    self.detection_queue,
                    self.model_path,
                    self.t1_start,
                    None,  # mp_stop_event no usado en hilos
                    self.is_process,
                ),
            ),
            threading.Thread(
                target=self.tracking_frames,
                args=(
                    self.detection_queue,
                    self.tracking_queue,
                    None,  # mp_stop_event no usado en hilos
                    self.is_process,
                ),
            ),
            threading.Thread(
                target=self.draw_and_write_frames,
                args=(
                    self.tracking_queue,
                    self.times_queue,
                    self.output_video_path,
                    self.CLASSES,
                    self.memory,
                    self.COLORS,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    None,  # mp_stop_event no usado en hilos
                    self.is_process,
                ),
            ),
            threading.Thread(
                target=self.write_to_csv,
                args=(
                    self.times_queue,
                    self.output_times,
                    self.parallel_mode,
                    self.stop_event,
                    None,  # mp_stop_event no usado en hilos
                    self.is_process,
                ),
            ),
            threading.Thread(
                target=self.hardware_usage,
                args=(
                    self.parallel_mode,
                    self.stop_event,
                    self.t1_start,
                    self.tcp_event,
                    self.is_tcp,
                    self.is_process,
                ),
            ),
        ]

        # Iniciar todos los hilos
        t1 = cv2.getTickCount()
        for thread in threads:
            thread.start()

        # Esperar a que se detenga el pipeline
        self.stop_event.wait()

        # Calcular tiempo total y FPS
        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        self.total_frames = self.get_total_frames(self.video_path)
        print(f"Total time: {time:.2f} s, FPS: {self.total_frames / time:.2f}")

        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join()

        print("Pipeline finished.")
