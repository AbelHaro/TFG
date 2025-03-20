"""
Módulo de visualización para el pipeline de detección y tracking.

Proporciona funciones para visualizar detecciones, tracking y métricas
en tiempo real y para generar visualizaciones para análisis posterior.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

from .types import (
    Frame,
    TrackingResult,
    ColorBGR,
    MetricsData
)
from .config import COLOR_MAPPING

class VisualizationManager:
    """
    Gestiona la visualización de resultados del pipeline.
    """

    def __init__(
        self,
        window_name: str = "Detection & Tracking",
        show_fps: bool = True,
        show_counts: bool = True,
        font_scale: float = 0.5,
        line_thickness: int = 2
    ):
        """
        Inicializa el gestor de visualización.

        Args:
            window_name: Nombre de la ventana de visualización.
            show_fps: Si se debe mostrar el FPS.
            show_counts: Si se debe mostrar el conteo de objetos.
            font_scale: Escala de la fuente para el texto.
            line_thickness: Grosor de las líneas para los bounding boxes.
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_counts = show_counts
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Inicializar contadores de objetos por clase
        self.class_counts: Dict[str, int] = {
            class_name: 0 for class_name in COLOR_MAPPING.keys()
        }

    def draw_detection_box(
        self,
        frame: Frame,
        box: Tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None,
        is_defective: bool = False
    ) -> Frame:
        """
        Dibuja un bounding box con información de detección/tracking.

        Args:
            frame: Frame sobre el que dibujar.
            box: Coordenadas del bounding box (x1, y1, x2, y2).
            class_name: Nombre de la clase detectada.
            confidence: Confianza de la detección.
            track_id: ID de tracking (opcional).
            is_defective: Si el objeto es defectuoso.

        Returns:
            Frame con el bounding box dibujado.
        """
        x1, y1, x2, y2 = box
        color = COLOR_MAPPING.get(
            f"{class_name}{'-d' if is_defective else ''}", 
            (255, 255, 255)
        )

        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # Preparar texto
        text = f"{class_name}"
        if track_id is not None:
            text = f"ID:{track_id} {text}"
        text += f" {confidence:.2f}"

        # Dibujar fondo para el texto
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.line_thickness)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1
        )

        # Dibujar texto
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.line_thickness
        )

        return frame

    def draw_metrics_overlay(
        self,
        frame: Frame,
        fps: float,
        frame_number: int,
        processing_time: float
    ) -> Frame:
        """
        Dibuja un overlay con métricas sobre el frame.

        Args:
            frame: Frame sobre el que dibujar.
            fps: FPS actual.
            frame_number: Número de frame.
            processing_time: Tiempo de procesamiento en ms.

        Returns:
            Frame con el overlay dibujado.
        """
        # Dibujar fondo semitransparente para métricas
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Dibujar métricas
        y_pos = 30
        if self.show_fps:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, y_pos),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.line_thickness
            )
            y_pos += 20

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, y_pos),
            self.font,
            self.font_scale,
            (0, 255, 0),
            self.line_thickness
        )
        y_pos += 20

        cv2.putText(
            frame,
            f"Time: {processing_time*1000:.1f}ms",
            (20, y_pos),
            self.font,
            self.font_scale,
            (0, 255, 0),
            self.line_thickness
        )

        return frame

    def draw_object_counts(self, frame: Frame) -> Frame:
        """
        Dibuja el conteo de objetos por clase.

        Args:
            frame: Frame sobre el que dibujar.

        Returns:
            Frame con el conteo dibujado.
        """
        if not self.show_counts:
            return frame

        # Dibujar fondo semitransparente para conteos
        overlay = frame.copy()
        height = len(self.class_counts) * 20 + 10
        cv2.rectangle(overlay, (frame.shape[1] - 150, 10), (frame.shape[1] - 10, height + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Dibujar conteos
        y_pos = 30
        for class_name, count in self.class_counts.items():
            color = COLOR_MAPPING.get(class_name, (255, 255, 255))
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (frame.shape[1] - 140, y_pos),
                self.font,
                self.font_scale,
                color,
                self.line_thickness
            )
            y_pos += 20

        return frame

    def update_display(
        self,
        frame: Frame,
        tracked_objects: List[TrackingResult],
        metrics: MetricsData
    ) -> None:
        """
        Actualiza la visualización con nuevos resultados.

        Args:
            frame: Frame actual.
            tracked_objects: Lista de objetos trackeados.
            metrics: Métricas actuales.
        """
        # Actualizar conteos
        self.class_counts = {class_name: 0 for class_name in COLOR_MAPPING.keys()}
        for obj in tracked_objects:
            class_name = obj.class_id  # Asumiendo que class_id es el nombre de la clase
            self.class_counts[class_name] += 1

        # Dibujar detecciones
        for obj in tracked_objects:
            self.draw_detection_box(
                frame,
                obj.box,
                obj.class_id,
                obj.confidence,
                obj.track_id,
                obj.is_defective
            )

        # Dibujar métricas y conteos
        frame = self.draw_metrics_overlay(
            frame,
            metrics.fps,
            0,  # frame_number se debería pasar como parte de metrics
            metrics.processing_time
        )
        frame = self.draw_object_counts(frame)

        # Mostrar frame
        cv2.imshow(self.window_name, frame)

    def create_performance_plot(
        self,
        metrics_data: List[MetricsData],
        output_path: str
    ) -> None:
        """
        Crea un gráfico de rendimiento.

        Args:
            metrics_data: Lista de métricas a graficar.
            output_path: Ruta donde guardar el gráfico.
        """
        plt.figure(figsize=(12, 8))
        
        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extraer datos
        frames = range(len(metrics_data))
        fps_data = [m.fps for m in metrics_data]
        proc_times = [m.processing_time * 1000 for m in metrics_data]  # Convertir a ms
        
        # Graficar FPS
        ax1.plot(frames, fps_data, 'b-', label='FPS')
        ax1.set_title('FPS over time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('FPS')
        ax1.grid(True)
        
        # Graficar tiempos de procesamiento
        ax2.plot(frames, proc_times, 'r-', label='Processing Time')
        ax2.set_title('Processing Time over time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Time (ms)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def cleanup(self) -> None:
        """
        Limpia los recursos de visualización.
        """
        cv2.destroyWindow(self.window_name)