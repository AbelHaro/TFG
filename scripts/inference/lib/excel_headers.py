"""
Configuración de las cabeceras para los archivos Excel generados durante la inferencia.
"""

# Cabeceras para tiempos de procesamiento
TIMING_HEADERS = [
    "frame_number",
    "capture_stage",
    "inference_stage",
    "tracking_stage",
    "writing_stage",
    "objects_count",
    "total_time(ms)",
    "time_per_object_tracking(ms)",
    "inference_stage_preprocess",
    "inference_stage_inference",
    "inference_stage_postprocess"
]

# Cabeceras para FPS
FPS_HEADERS = [
    "Frame",
    "FPS"
]

# Cabeceras para uso de hardware
HARDWARE_HEADERS = [
    "Timestamp",
    "RAM Usage (%)",
    "CPU Usage (%)",
    "GPU Usage (%)",
    "Temperature CPU (°C)",
    "Temperature GPU (°C)"
]