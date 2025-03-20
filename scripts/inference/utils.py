"""
Módulo de utilidades para el pipeline de detección y tracking.

Este módulo contiene funciones auxiliares utilizadas en diferentes partes del pipeline,
facilitando la reutilización de código y mejorando su mantenimiento.
"""

import cv2
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from argparse import Namespace

def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Configura el sistema de logging con un formato consistente.

    Args:
        level: Nivel de logging a utilizar. Por defecto es DEBUG.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def get_video_info(video_path: str) -> Tuple[int, Tuple[int, int]]:
    """
    Obtiene información básica de un archivo de video.

    Args:
        video_path: Ruta al archivo de video.

    Returns:
        Tupla con (total_frames, (height, width))

    Raises:
        IOError: Si no se puede abrir el archivo de video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return total_frames, (height, width)

def ensure_dir_exists(path: str) -> None:
    """
    Asegura que un directorio existe, creándolo si es necesario.

    Args:
        path: Ruta del directorio a verificar/crear.
    """
    os.makedirs(path, exist_ok=True)

def format_detection_results(results: Any) -> Namespace:
    """
    Formatea los resultados de detección en un formato consistente.

    Args:
        results: Resultados de detección del modelo.

    Returns:
        Namespace con los resultados formateados (xywh, conf, cls).
    """
    return Namespace(
        xywh=results[0].boxes.xywh.cpu(),
        conf=results[0].boxes.conf.cpu(),
        cls=results[0].boxes.cls.cpu(),
    )

def format_sahi_results(
    xywh: List[List[float]], 
    confidences: List[float], 
    classes: List[float]
) -> Namespace:
    """
    Formatea los resultados de SAHI en el mismo formato que los resultados normales.

    Args:
        xywh: Lista de coordenadas [x, y, width, height].
        confidences: Lista de valores de confianza.
        classes: Lista de clases predichas.

    Returns:
        Namespace con los resultados formateados.
    """
    return Namespace(
        xywh=torch.tensor(xywh, dtype=torch.float32) if xywh else torch.empty((0, 4), dtype=torch.float32),
        conf=torch.tensor(confidences, dtype=torch.float32) if confidences else torch.empty(0, dtype=torch.float32),
        cls=torch.tensor(classes, dtype=torch.float32) if classes else torch.empty(0, dtype=torch.float32)
    )

def calculate_time(t1: int, t2: int) -> float:
    """
    Calcula el tiempo transcurrido entre dos marcas de tiempo.

    Args:
        t1: Marca de tiempo inicial.
        t2: Marca de tiempo final.

    Returns:
        Tiempo transcurrido en segundos.
    """
    return (t2 - t1) / cv2.getTickFrequency()

def update_tracking_memory(
    tracked_objects: List[Any],
    memory: Dict[int, Dict[str, Any]],
    class_mapping: Dict[int, str],
    frame_age: int
) -> None:
    """
    Actualiza la memoria de tracking con los objetos detectados.

    Args:
        tracked_objects: Lista de objetos trackeados.
        memory: Diccionario de memoria de tracking.
        class_mapping: Mapeo de índices de clase a nombres.
        frame_age: Número de frames que un objeto permanece en memoria.
    """
    # Actualizar objetos existentes
    for obj in tracked_objects:
        track_id = int(obj[4])
        detected_class = class_mapping[int(obj[6])]
        
        is_defective = detected_class.endswith("-d")
        if track_id in memory:
            entry = memory[track_id]
            entry["defective"] |= is_defective
            entry["visible_frames"] = frame_age
            if entry["defective"] and not is_defective:
                detected_class += "-d"
            entry["class"] = detected_class
        else:
            memory[track_id] = {
                "defective": is_defective,
                "visible_frames": frame_age,
                "class": detected_class,
            }

    # Actualizar contadores de frames y eliminar objetos antiguos
    for track_id in list(memory):
        memory[track_id]["visible_frames"] -= 1
        if memory[track_id]["visible_frames"] <= 0:
            del memory[track_id]

def draw_detection_info(
    frame: np.ndarray,
    tracked_objects: List[Any],
    memory: Dict[int, Dict[str, Any]],
    colors: Dict[str, Tuple[int, int, int]],
    min_confidence: float = 0.4,
    frame_number: Optional[int] = None
) -> np.ndarray:
    """
    Dibuja la información de detección en el frame.

    Args:
        frame: Frame sobre el que dibujar.
        tracked_objects: Lista de objetos trackeados.
        memory: Diccionario de memoria de tracking.
        colors: Mapeo de clases a colores.
        min_confidence: Confianza mínima para mostrar detecciones.
        frame_number: Número de frame actual (opcional).

    Returns:
        Frame con la información dibujada.
    """
    frame_copy = frame.copy()

    # Dibujar objetos trackeados
    for obj in tracked_objects:
        xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
        conf = float(obj[5])

        if conf < min_confidence:
            continue

        detected_class = memory[obj_id]["class"]
        color = colors.get(detected_class, (255, 255, 255))
        
        # Dibujar rectángulo
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Dibujar texto
        text = f"ID:{obj_id} {detected_class} {conf:.2f}"
        cv2.putText(
            frame_copy,
            text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # Dibujar número de frame si se proporciona
    if frame_number is not None:
        cv2.putText(
            frame_copy,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    return frame_copy