"""
Módulo de validación para el pipeline de detección y tracking.

Proporciona funciones para validar parámetros, configuraciones y datos
de entrada antes de su uso en el pipeline.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch

from .types import (
    Frame,
    ModelPath,
    VideoPath,
    HardwareConfig,
    ProcessingConfig,
    PipelineConfig,
    Resolution
)
from .exceptions import (
    InvalidConfigError,
    VideoError,
    ModelError
)

def validate_video_path(video_path: VideoPath) -> None:
    """
    Valida que el archivo de video existe y es accesible.

    Args:
        video_path: Ruta al archivo de video.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        VideoError: Si el archivo no es un video válido.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encuentra el archivo de video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"No se puede abrir el archivo de video: {video_path}")
    
    # Verificar que el video tiene frames
    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise VideoError(f"El archivo de video está vacío o corrupto: {video_path}")
    
    cap.release()

def validate_model_path(model_path: ModelPath) -> None:
    """
    Valida que el archivo del modelo existe y es accesible.

    Args:
        model_path: Ruta al archivo del modelo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ModelError: Si el archivo no es un modelo válido.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el archivo del modelo: {model_path}")
    
    # Verificar extensión del modelo
    valid_extensions = ['.engine', '.trt', '.onnx', '.pt']
    if not any(model_path.endswith(ext) for ext in valid_extensions):
        raise ModelError(
            f"Formato de modelo no soportado. Debe ser uno de: {', '.join(valid_extensions)}"
        )

def validate_frame(frame: Frame) -> None:
    """
    Valida que un frame tiene el formato correcto.

    Args:
        frame: Frame a validar.

    Raises:
        ValueError: Si el frame no tiene el formato correcto.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("El frame debe ser un numpy.ndarray")
    
    if len(frame.shape) != 3:
        raise ValueError("El frame debe tener 3 dimensiones (altura, anchura, canales)")
    
    if frame.shape[2] != 3:
        raise ValueError("El frame debe tener 3 canales (BGR)")
    
    if frame.dtype != np.uint8:
        raise ValueError("El frame debe ser de tipo uint8")

def validate_hardware_config(config: HardwareConfig) -> None:
    """
    Valida la configuración de hardware.

    Args:
        config: Configuración de hardware a validar.

    Raises:
        InvalidConfigError: Si la configuración no es válida.
    """
    valid_devices = ['cuda', 'cpu', 'dla']
    if config.device not in valid_devices:
        raise InvalidConfigError(
            'device',
            config.device,
            f"Dispositivo no válido. Debe ser uno de: {', '.join(valid_devices)}"
        )

    valid_precision = ['FP32', 'FP16', 'INT8']
    if config.precision not in valid_precision:
        raise InvalidConfigError(
            'precision',
            config.precision,
            f"Precisión no válida. Debe ser una de: {', '.join(valid_precision)}"
        )

    if config.batch_size < 1:
        raise InvalidConfigError(
            'batch_size',
            str(config.batch_size),
            "El tamaño del batch debe ser mayor que 0"
        )

    if config.device == 'dla' and config.dla_core not in [0, 1]:
        raise InvalidConfigError(
            'dla_core',
            str(config.dla_core),
            "El core DLA debe ser 0 o 1"
        )

def validate_processing_config(config: ProcessingConfig) -> None:
    """
    Valida la configuración de procesamiento.

    Args:
        config: Configuración de procesamiento a validar.

    Raises:
        InvalidConfigError: Si la configuración no es válida.
    """
    if config.num_workers < 1:
        raise InvalidConfigError(
            'num_workers',
            str(config.num_workers),
            "El número de workers debe ser mayor que 0"
        )

    if config.queue_size < 1:
        raise InvalidConfigError(
            'queue_size',
            str(config.queue_size),
            "El tamaño de la cola debe ser mayor que 0"
        )

def validate_pipeline_config(config: PipelineConfig) -> None:
    """
    Valida la configuración completa del pipeline.

    Args:
        config: Configuración del pipeline a validar.

    Raises:
        InvalidConfigError: Si la configuración no es válida.
    """
    validate_hardware_config(config.hardware)
    validate_processing_config(config.processing)
    validate_video_path(config.video_path)
    validate_model_path(config.model_path)

def validate_resolution(resolution: Resolution) -> None:
    """
    Valida una resolución de imagen.

    Args:
        resolution: Tupla (width, height) a validar.

    Raises:
        ValueError: Si la resolución no es válida.
    """
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError("La resolución debe tener valores positivos")
    
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("La resolución debe tener valores pares")

def validate_detection_threshold(threshold: float) -> None:
    """
    Valida un umbral de detección.

    Args:
        threshold: Umbral a validar.

    Raises:
        ValueError: Si el umbral no es válido.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral de detección debe estar entre 0 y 1")

def validate_model_input(
    frame: Frame,
    expected_size: Tuple[int, int],
    expected_channels: int = 3
) -> None:
    """
    Valida que un frame cumple con los requisitos de entrada del modelo.

    Args:
        frame: Frame a validar.
        expected_size: Tamaño esperado (width, height).
        expected_channels: Número de canales esperado.

    Raises:
        ValueError: Si el frame no cumple con los requisitos.
    """
    if frame.shape[:2] != expected_size[::-1]:  # OpenCV usa (height, width)
        raise ValueError(
            f"Tamaño de frame incorrecto. Esperado {expected_size}, recibido {frame.shape[:2][::-1]}"
        )
    
    if frame.shape[2] != expected_channels:
        raise ValueError(
            f"Número de canales incorrecto. Esperado {expected_channels}, recibido {frame.shape[2]}"
        )

def validate_cuda_available() -> None:
    """
    Valida que CUDA está disponible si se intenta usar.

    Raises:
        RuntimeError: Si CUDA no está disponible.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Asegúrate de tener instalados los drivers de NVIDIA"
        )

def validate_dla_available(core_id: int) -> None:
    """
    Valida que un core DLA específico está disponible.

    Args:
        core_id: ID del core DLA a validar.

    Raises:
        RuntimeError: Si el core DLA no está disponible.
    """
    # Esta es una implementación simplificada. En un sistema real,
    # necesitarías verificar la disponibilidad real del DLA.
    if core_id not in [0, 1]:
        raise RuntimeError(f"Core DLA {core_id} no válido. Debe ser 0 o 1")