"""
Clase base para el pipeline de detección y tracking.

Esta clase implementa la funcionalidad base compartida por todas las variantes
del pipeline (hilos, multiprocesos, hardware múltiple, etc.).
"""

from abc import ABC, abstractmethod
import cv2
import logging
import os
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Union, Optional, Any
from argparse import Namespace

from .config import (
    CLASS_MAPPING,
    COLOR_MAPPING,
    TRACKING_CONFIG,
    VIDEO_CONFIG,
    TCP_CONFIG
)
from .utils import (
    setup_logging,
    get_video_info,
    ensure_dir_exists,
    format_detection_results,
    calculate_time,
    update_tracking_memory,
    draw_detection_info
)
from .exceptions import (
    VideoOpenError,
    VideoWriteError,
    ModelLoadError,
    InferenceError,
    TCPConnectionError
)
from .classes.shared_circular_buffer import SharedCircularBuffer
from .classes.tracker_wrapper import TrackerWrapper


class BasePipeline(ABC):
    """
    Clase base abstracta que define la interfaz y funcionalidad común
    para todos los pipelines de detección y tracking.
    """

    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_video_path: str,
        output_times: str,
        parallel_mode: str,
        use_tcp: bool = False
    ):
        """
        Inicializa el pipeline base.

        Args:
            video_path: Ruta al video de entrada.
            model_path: Ruta al modelo de detección.
            output_video_path: Ruta donde guardar el video procesado.
            output_times: Prefijo para los archivos de tiempos.
            parallel_mode: Modo de paralelización ('threads', 'mp', etc.).
            use_tcp: Si se debe usar comunicación TCP.

        Raises:
            VideoOpenError: Si no se puede abrir el video de entrada.
            FileNotFoundError: Si no existe el archivo de video.
        """
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.use_tcp = use_tcp

        # Configurar logging
        setup_logging()
        self.logger = logging.getLogger(__name__)

        # Validar archivo de video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No se encuentra el archivo de video: {video_path}")
        
        # Obtener información del video
        try:
            self.total_frames, (self.frame_height, self.frame_width) = get_video_info(video_path)
        except IOError as e:
            raise VideoOpenError(video_path) from e

        # Crear directorios necesarios
        ensure_dir_exists(os.path.dirname(output_video_path))

        # Inicializar estado compartido
        self.memory: Dict[int, Dict[str, Any]] = {}
        self.stop_event = mp.Event()
        self.tcp_event = mp.Event()

    def _initialize_video_writer(self) -> cv2.VideoWriter:
        """
        Inicializa el escritor de video.

        Returns:
            VideoWriter configurado.

        Raises:
            VideoWriteError: Si no se puede crear el escritor de video.
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["CODEC"])
            return cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                VIDEO_CONFIG["FPS"],
                (self.frame_width, self.frame_height)
            )
        except Exception as e:
            raise VideoWriteError(self.output_video_path) from e

    def _setup_tcp_server(self) -> Tuple[Any, Any]:
        """
        Configura el servidor TCP si está habilitado.

        Returns:
            Tupla (socket_cliente, socket_servidor)

        Raises:
            TCPConnectionError: Si hay un error al configurar el servidor TCP.
        """
        if not self.use_tcp:
            return None, None

        try:
            from .lib.tcp import tcp_server
            return tcp_server(TCP_CONFIG["HOST"], TCP_CONFIG["PORT"])
        except Exception as e:
            raise TCPConnectionError(TCP_CONFIG["HOST"], TCP_CONFIG["PORT"]) from e

    def capture_frames(
        self,
        frame_queue: Union[mp.Queue, SharedCircularBuffer],
        stop_event: mp.Event,
        tcp_event: mp.Event,
        is_tcp: bool,
        mp_stop_event: Optional[mp.Event] = None,
        mh_num: int = 1,
        is_process: bool = False
    ) -> None:
        """
        Captura frames del video y los coloca en la cola.

        Args:
            frame_queue: Cola donde colocar los frames capturados.
            stop_event: Evento para detener la captura.
            tcp_event: Evento para sincronización TCP.
            is_tcp: Si se está usando TCP.
            mp_stop_event: Evento para sincronización de multiprocesos.
            mh_num: Número de hardware para multihardware.
            is_process: Si se está ejecutando como proceso.
        """
        self.logger.debug("Iniciando captura de frames")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            for _ in range(mh_num):
                frame_queue.put(None)
            raise VideoOpenError(self.video_path)

        if is_tcp:
            tcp_event.wait()

        frame_count = 0
        while cap.isOpened() and not stop_event.is_set():
            t1 = cv2.getTickCount()
            ret, frame = cap.read()

            if not ret:
                self.logger.debug(f"Fin de video alcanzado. Frames procesados: {frame_count}")
                break

            t2 = cv2.getTickCount()
            times = {"capture": calculate_time(t1, t2)}
            frame_queue.put((frame, times))
            frame_count += 1

        cap.release()
        for _ in range(mh_num):
            frame_queue.put(None)

        if mp_stop_event:
            mp_stop_event.wait()
        if is_process:
            os._exit(0)

    @abstractmethod
    def run(self) -> None:
        """
        Ejecuta el pipeline completo. Debe ser implementado por las clases derivadas.
        """
        pass

    def cleanup(self) -> None:
        """
        Realiza la limpieza necesaria al finalizar el pipeline.
        """
        cv2.destroyAllWindows()
        self.stop_event.set()