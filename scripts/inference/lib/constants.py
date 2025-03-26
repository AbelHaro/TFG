"""
Configuración de constantes utilizadas en el pipeline de detección y tracking.
"""

# Nombres de campos para tiempos
TIMING_FIELDS = {
    "CAPTURE": "capture",
    "PROCESSING": "processing",
    "TRACKING": "tracking",
    "WRITING": "writing",
    "OBJECTS_COUNT": "objects_count",
    "DETECT_FUNCTION": "detect_function",
    "PREPROCESS": "preprocess",
    "INFERENCE": "inference",
    "POSTPROCESS": "postprocess"
}

# Nombres de hojas en Excel
EXCEL_SHEETS = {
    "TIMES": "Times",
    "FPS": "FPS",
    "HARDWARE": "Hardware Usage"
}