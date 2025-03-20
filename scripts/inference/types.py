"""
Tipos personalizados para el pipeline de detección y tracking.

Define tipos personalizados y alias de tipo para mejorar la claridad del código
y facilitar el tipado estático.
"""

from typing import Dict, List, Tuple, Union, TypeVar, NamedTuple, Protocol, runtime_checkable
import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as mp
from .classes.shared_circular_buffer import SharedCircularBuffer

# Tipos básicos
Frame = np.ndarray  # Imagen en formato numpy array
ModelPath = str     # Ruta a un archivo de modelo
VideoPath = str     # Ruta a un archivo de video

# Tipos para resultados de detección
class DetectionResult(NamedTuple):
    """Resultado de una detección individual."""
    box: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float
    class_id: int

class BatchDetectionResult(NamedTuple):
    """Resultado de una batch de detecciones."""
    boxes: Tensor        # Shape: (N, 4) - XYWH format
    confidences: Tensor  # Shape: (N,)
    class_ids: Tensor    # Shape: (N,)

# Tipos para tracking
class TrackingResult(NamedTuple):
    """Resultado de tracking para un objeto."""
    track_id: int
    box: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float
    class_id: int
    is_defective: bool

# Tipos para métricas y tiempos
class TimingInfo(Dict[str, float]):
    """Información de tiempos de procesamiento."""
    pass

class MetricsData(NamedTuple):
    """Datos de métricas recolectados durante el procesamiento."""
    fps: float
    processing_time: float
    detection_time: float
    tracking_time: float
    total_objects: int

# Tipos para colas
QueueType = Union[mp.Queue, SharedCircularBuffer]
QueueItem = Tuple[Frame, Union[BatchDetectionResult, List[TrackingResult]], TimingInfo]

# Tipos para configuración
class HardwareConfig(NamedTuple):
    """Configuración de hardware."""
    device: str
    precision: str
    batch_size: int
    dla_core: int

class ProcessingConfig(NamedTuple):
    """Configuración de procesamiento."""
    num_workers: int
    queue_size: int
    use_multiprocessing: bool
    use_shared_memory: bool

# Protocolos
@runtime_checkable
class ModelProtocol(Protocol):
    """Protocolo que define la interfaz esperada para modelos de detección."""
    
    def __call__(self, frame: Frame) -> BatchDetectionResult:
        """Realiza la inferencia en un frame."""
        ...

    def to(self, device: str) -> 'ModelProtocol':
        """Mueve el modelo a un dispositivo específico."""
        ...

@runtime_checkable
class TrackerProtocol(Protocol):
    """Protocolo que define la interfaz esperada para trackers."""
    
    def update(self, detections: BatchDetectionResult, frame: Frame) -> List[TrackingResult]:
        """Actualiza el tracking con nuevas detecciones."""
        ...

    def reset(self) -> None:
        """Reinicia el estado del tracker."""
        ...

# Tipos genéricos
T = TypeVar('T')
QueueContent = TypeVar('QueueContent')

# Alias de tipos comunes
Boxes = List[Tuple[float, float, float, float]]
ClassIDs = List[int]
Confidences = List[float]
ColorBGR = Tuple[int, int, int]
Resolution = Tuple[int, int]  # width, height

# Tipos para memoria de tracking
TrackingMemory = Dict[int, Dict[str, Union[bool, int, str]]]

# Tipos para eventos
EventType = mp.Event

# Tipos para configuración del pipeline
class PipelineConfig(NamedTuple):
    """Configuración completa del pipeline."""
    hardware: HardwareConfig
    processing: ProcessingConfig
    model_path: ModelPath
    video_path: VideoPath
    output_path: VideoPath
    use_tcp: bool