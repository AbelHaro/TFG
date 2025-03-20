"""
Módulo para el manejo de métricas y telemetría del pipeline.

Proporciona funcionalidades para recolectar, agregar y exportar métricas
de rendimiento y estadísticas del pipeline de detección y tracking.
"""

import csv
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from .types import MetricsData, TimingInfo
from .exceptions import FileWriteError
from .config import FILE_PATHS

@dataclass
class FrameMetrics:
    """Métricas para un frame individual."""
    frame_number: int
    capture_time: float
    preprocessing_time: float
    inference_time: float
    postprocessing_time: float
    tracking_time: float
    visualization_time: float
    total_time: float
    num_detections: int
    num_tracked: int
    fps: float

@dataclass
class BatchMetrics:
    """Métricas para un batch de frames."""
    batch_size: int
    batch_inference_time: float
    avg_inference_time: float
    fps: float
    memory_usage: float
    gpu_utilization: float

@dataclass
class HardwareMetrics:
    """Métricas de uso de hardware."""
    cpu_usage: float
    gpu_usage: float
    gpu_memory: float
    ram_usage: float
    temperature: float
    power_usage: float

class MetricsCollector:
    """
    Recolector de métricas para el pipeline de detección y tracking.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Inicializa el recolector de métricas.

        Args:
            output_dir: Directorio donde guardar las métricas.
            experiment_name: Nombre del experimento actual.
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Crear directorios necesarios
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar estructuras de datos
        self.frame_metrics: List[FrameMetrics] = []
        self.batch_metrics: List[BatchMetrics] = []
        self.hardware_metrics: List[HardwareMetrics] = []
        self.running_stats = defaultdict(list)
        
        # Timestamp de inicio
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_frame_metrics(self, metrics: FrameMetrics) -> None:
        """Añade métricas de un frame."""
        self.frame_metrics.append(metrics)

    def add_batch_metrics(self, metrics: BatchMetrics) -> None:
        """Añade métricas de un batch."""
        self.batch_metrics.append(metrics)

    def add_hardware_metrics(self, metrics: HardwareMetrics) -> None:
        """Añade métricas de hardware."""
        self.hardware_metrics.append(metrics)

    def update_running_stats(self, name: str, value: float) -> None:
        """
        Actualiza estadísticas en tiempo real.

        Args:
            name: Nombre de la métrica.
            value: Valor a añadir.
        """
        self.running_stats[name].append(value)

    def get_running_average(self, name: str, window_size: int = 100) -> float:
        """
        Calcula la media móvil de una métrica.

        Args:
            name: Nombre de la métrica.
            window_size: Tamaño de la ventana para la media móvil.

        Returns:
            Media móvil de la métrica.
        """
        values = self.running_stats[name]
        if not values:
            return 0.0
        return np.mean(values[-window_size:])

    def export_metrics(self) -> None:
        """
        Exporta todas las métricas recolectadas a archivos CSV y JSON.
        """
        try:
            # Crear nombre base para los archivos
            base_filename = f"{self.experiment_name}_{self.timestamp}"
            
            # Exportar métricas de frames
            if self.frame_metrics:
                df_frames = pd.DataFrame([asdict(m) for m in self.frame_metrics])
                df_frames.to_csv(
                    os.path.join(self.output_dir, f"{base_filename}_frame_metrics.csv"),
                    index=False
                )

            # Exportar métricas de batches
            if self.batch_metrics:
                df_batches = pd.DataFrame([asdict(m) for m in self.batch_metrics])
                df_batches.to_csv(
                    os.path.join(self.output_dir, f"{base_filename}_batch_metrics.csv"),
                    index=False
                )

            # Exportar métricas de hardware
            if self.hardware_metrics:
                df_hardware = pd.DataFrame([asdict(m) for m in self.hardware_metrics])
                df_hardware.to_csv(
                    os.path.join(self.output_dir, f"{base_filename}_hardware_metrics.csv"),
                    index=False
                )

            # Calcular y exportar resumen de métricas
            summary = self._generate_summary()
            with open(os.path.join(self.output_dir, f"{base_filename}_summary.json"), 'w') as f:
                json.dump(summary, f, indent=4)

        except Exception as e:
            raise FileWriteError(self.output_dir) from e

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de todas las métricas recolectadas.

        Returns:
            Diccionario con el resumen de métricas.
        """
        total_time = time.time() - self.start_time
        
        # Calcular estadísticas de frames
        frame_stats = {
            "total_frames": len(self.frame_metrics),
            "avg_fps": len(self.frame_metrics) / total_time if total_time > 0 else 0,
            "avg_inference_time": np.mean([m.inference_time for m in self.frame_metrics]) if self.frame_metrics else 0,
            "avg_tracking_time": np.mean([m.tracking_time for m in self.frame_metrics]) if self.frame_metrics else 0,
            "avg_detections": np.mean([m.num_detections for m in self.frame_metrics]) if self.frame_metrics else 0
        }

        # Calcular estadísticas de hardware
        hardware_stats = {
            "avg_cpu_usage": np.mean([m.cpu_usage for m in self.hardware_metrics]) if self.hardware_metrics else 0,
            "avg_gpu_usage": np.mean([m.gpu_usage for m in self.hardware_metrics]) if self.hardware_metrics else 0,
            "avg_gpu_memory": np.mean([m.gpu_memory for m in self.hardware_metrics]) if self.hardware_metrics else 0,
            "avg_temperature": np.mean([m.temperature for m in self.hardware_metrics]) if self.hardware_metrics else 0
        }

        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "total_time": total_time,
            "frame_statistics": frame_stats,
            "hardware_statistics": hardware_stats,
            "runtime_parameters": {
                "total_frames_processed": len(self.frame_metrics),
                "total_batches_processed": len(self.batch_metrics)
            }
        }

    def log_metrics(self) -> None:
        """
        Registra las métricas actuales en el logger.
        """
        current_fps = self.get_running_average("fps")
        avg_inference_time = self.get_running_average("inference_time")
        avg_tracking_time = self.get_running_average("tracking_time")

        self.logger.info(
            f"FPS: {current_fps:.2f} | "
            f"Inference: {avg_inference_time*1000:.1f}ms | "
            f"Tracking: {avg_tracking_time*1000:.1f}ms"
        )

    def create_excel_report(self) -> None:
        """
        Crea un informe detallado en Excel con todas las métricas.
        """
        try:
            filename = os.path.join(
                self.output_dir,
                f"{self.experiment_name}_{self.timestamp}_report.xlsx"
            )

            with pd.ExcelWriter(filename) as writer:
                # Escribir métricas de frames
                if self.frame_metrics:
                    pd.DataFrame([asdict(m) for m in self.frame_metrics]).to_excel(
                        writer, sheet_name='Frame Metrics', index=False
                    )

                # Escribir métricas de batches
                if self.batch_metrics:
                    pd.DataFrame([asdict(m) for m in self.batch_metrics]).to_excel(
                        writer, sheet_name='Batch Metrics', index=False
                    )

                # Escribir métricas de hardware
                if self.hardware_metrics:
                    pd.DataFrame([asdict(m) for m in self.hardware_metrics]).to_excel(
                        writer, sheet_name='Hardware Metrics', index=False
                    )

                # Escribir resumen
                summary = pd.DataFrame([self._generate_summary()])
                summary.to_excel(writer, sheet_name='Summary', index=False)

        except Exception as e:
            raise FileWriteError(filename) from e