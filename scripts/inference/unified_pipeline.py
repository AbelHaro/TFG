from abc import ABC
import cv2
import torch.multiprocessing as mp
from queue import Queue
from classes.shared_circular_buffer import SharedCircularBuffer
from detection_tracking_pipeline import DetectionTrackingPipeline
import threading
import logging
from typing import Union, List, Type, Any


class UnifiedPipeline(DetectionTrackingPipeline):
    """Pipeline unificado que soporta diferentes estrategias de paralelización."""

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
        dla0_model: str = None,
        dla1_model: str = None,
    ):
        """Inicializa el pipeline unificado."""
        self.parallel_mode = parallel_mode
        self.is_process = parallel_mode != "threads"
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.is_tcp = is_tcp
        self.sahi = sahi
        self.max_fps = max_fps
        self.dla0_model = dla0_model
        self.dla1_model = dla1_model
        self.mh_num = 1
        self._initialize_queues()
        self._initialize_events()
        self.memory = {}

    def _initialize_queues(self):
        """Inicializa las colas según el modo de paralelización."""
        queue_size = 1 if self.max_fps else 10

        if self.parallel_mode == "mp_hardware":
            # Configuración para múltiple hardware

            self.frame_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.detection_queue_GPU = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.detection_queue_DLA0 = SharedCircularBuffer(
                queue_size=queue_size, max_item_size=16
            )
            self.detection_queue_DLA1 = SharedCircularBuffer(
                queue_size=queue_size, max_item_size=16
            )
            self.tracking_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.times_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.mh_num = 3  # GPU + 2 DLA
        elif self.parallel_mode == "threads":
            # Configuración para hilos
            self.frame_queue = Queue(maxsize=queue_size)
            self.detection_queue = Queue(maxsize=queue_size)
            self.tracking_queue = Queue(maxsize=queue_size)
            self.times_queue = Queue(maxsize=queue_size)
        elif self.parallel_mode == "mp_shared_memory":
            # Configuración para memoria compartida
            self.frame_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.detection_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.tracking_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.times_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
        else:
            # Configuración para multiprocesos estándar
            self.frame_queue = (
                mp.Queue(maxsize=queue_size) if not self.max_fps else mp.Queue(maxsize=1)
            )
            self.detection_queue = mp.Queue(maxsize=queue_size)
            self.tracking_queue = mp.Queue(maxsize=queue_size)
            self.times_queue = mp.Queue(maxsize=queue_size)

    def _initialize_events(self):
        # self.stop_event = mp.Event() if self.is_process else threading.Event()
        # self.t1_start = mp.Event() if self.is_process else threading.Event()
        # self.tcp_event = mp.Event() if self.is_process else threading.Event()
        # self.mp_stop_event = mp.Event() if self.is_process else threading.Event()
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()
        self.tcp_event = mp.Event()
        self.mp_stop_event = mp.Event()

    def _get_worker_class(self) -> Type[Union[mp.Process, threading.Thread]]:
        """Retorna la clase de worker según el modo de paralelización."""
        return threading.Thread if self.parallel_mode == "threads" else mp.Process

    def _create_workers(self) -> List[Union[mp.Process, threading.Thread]]:
        """Crea los workers para el pipeline."""
        Worker = self._get_worker_class()
        workers = []

        # Worker común para captura de frames
        workers.append(
            Worker(
                target=self.capture_frames,
                args=(
                    self.video_path,
                    self.frame_queue,
                    self.t1_start,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    self.mp_stop_event,
                    self.mh_num,
                    self.is_process,
                    self.max_fps,
                ),
            )
        )

        # Workers específicos según el modo
        if self.parallel_mode == "mp_hardware":
            # Workers para procesamiento en múltiple hardware
            for model_path, detection_queue in [
                (self.model_path, self.detection_queue_GPU),
                (self.dla0_model, self.detection_queue_DLA0),
                (self.dla1_model, self.detection_queue_DLA1),
            ]:
                if model_path:
                    workers.append(
                        Worker(
                            target=self.process_frames_sahi if self.sahi else self.process_frames,
                            args=(
                                self.frame_queue,
                                detection_queue,
                                model_path,
                                self.t1_start,
                                self.mp_stop_event,
                                self.is_process,
                            ),
                        )
                    )

            # Worker específico para tracking multihardware
            workers.append(
                Worker(
                    target=self.tracking_frames_multihardware,
                    args=(
                        self.detection_queue_GPU,
                        self.detection_queue_DLA0,
                        self.detection_queue_DLA1,
                        self.tracking_queue,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )
        else:
            # Worker estándar para procesamiento
            workers.append(
                Worker(
                    target=self.process_frames_sahi if self.sahi else self.process_frames,
                    args=(
                        self.frame_queue,
                        self.detection_queue,
                        self.model_path,
                        self.t1_start,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )

            # Worker estándar para tracking
            workers.append(
                Worker(
                    target=self.tracking_frames,
                    args=(
                        self.detection_queue,
                        self.tracking_queue,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )

        # Workers comunes para todos los modos
        workers.extend(
            [
                Worker(
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
                        self.mp_stop_event,
                        self.is_process,
                    ),
                ),
                Worker(
                    target=self.write_to_csv,
                    args=(
                        self.times_queue,
                        self.output_times,
                        self.parallel_mode,
                        self.t1_start,
                        self.stop_event,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                ),
                Worker(
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
        )
        return workers

    def _cleanup(self):
        """Limpia recursos según el modo de paralelización."""
        # Limpiar las colas según el modo
        if self.parallel_mode in ["mp_shared_memory", "mp_hardware"]:
            queues = [self.frame_queue, self.tracking_queue, self.times_queue]

            if self.parallel_mode == "mp_hardware":
                queues.extend(
                    [
                        self.detection_queue_GPU,
                        self.detection_queue_DLA0,
                        self.detection_queue_DLA1,
                    ]
                )
            else:
                queues.append(self.detection_queue)

            for queue in queues:
                queue.close()
                queue.unlink()

    def run(self):
        """Ejecuta el pipeline unificado."""
        # Crear e iniciar workers
        workers = self._create_workers()

        for worker in workers:
            worker.start()

        self.t1_start.wait()

        t1 = cv2.getTickCount()
        # Esperar a que termine el pipeline
        self.stop_event.wait()

        # Calcular estadísticas finales
        t2 = cv2.getTickCount()
        # Limpiar recursos
        self._cleanup()

        time = (t2 - t1) / cv2.getTickFrequency()
        self.total_frames = self.get_total_frames(self.video_path)
        print(f"Total time: {time:.2f} s, FPS: {self.total_frames / time:.2f}")

        # Esperar a que terminen los hilos si es necesario
        if self.parallel_mode == "threads":
            for worker in workers:
                print(f"Waiting for {worker.name} to finish...")
                worker.join()
                print(f"{worker.name} finished.")

        print("Pipeline finished.")
