"""
Módulo de configuración para el pipeline de detección y tracking.

Este módulo contiene todas las constantes y configuraciones utilizadas en el pipeline,
centralizando los valores para facilitar su mantenimiento y modificación.
"""

from typing import Dict, Tuple

# Configuración de clases y colores
CLASS_MAPPING: Dict[int, str] = {
    0: "negra",
    1: "blanca",
    2: "verde",
    3: "azul",
    4: "negra-d",
    5: "blanca-d",
    6: "verde-d",
    7: "azul-d",
}

COLOR_MAPPING: Dict[str, Tuple[int, int, int]] = {
    "negra": (0, 0, 255),      # Rojo en formato BGR
    "blanca": (0, 255, 0),     # Verde en formato BGR
    "verde": (255, 0, 0),      # Azul en formato BGR
    "azul": (255, 255, 0),     # Cyan en formato BGR
    "negra-d": (0, 165, 255),  # Naranja en formato BGR
    "blanca-d": (255, 165, 0), # Azul claro en formato BGR
    "verde-d": (255, 105, 180),# Rosa en formato BGR
    "azul-d": (255, 0, 255),   # Magenta en formato BGR
}

# Configuración del tracking
TRACKING_CONFIG = {
    "FRAME_RATE": 30,
    "MEMORY_FRAME_AGE": 60,    # Número de frames que un objeto permanece en memoria
    "MIN_CONFIDENCE": 0.4,     # Confianza mínima para mostrar detecciones
}

# Configuración de video
VIDEO_CONFIG = {
    "CODEC": "mp4v",
    "FPS": 30,
}

# Configuración de la inferencia SAHI
SAHI_CONFIG = {
    "NEW_WIDTH": 640,
    "NEW_HEIGHT": 640,
    "OVERLAP_PIXELS": 100,
    "CONFIDENCE_THRESHOLD": 0.5,
    "IOU_THRESHOLD": 0.4,
    "OVERLAP_THRESHOLD": 0.8,
    "BATCH_SIZE": 4,
}

# Configuración de TCP
TCP_CONFIG = {
    "HOST": "0.0.0.0",
    "PORT": 8765,
}

# Configuración de hardware
HARDWARE_CONFIG = {
    "TEGRASTATS_INTERVAL": 100,  # Intervalo de muestreo en ms
}

# Rutas de archivos
FILE_PATHS = {
    "BASE_EXCEL_PATH": "/TFG/excels",
    "HARDWARE_STATS_PATH": "aux_files/hardware_usage.txt",
    "HARDWARE_USAGE_CSV": "aux_files/hardware_usage_aux.csv",
}