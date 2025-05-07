import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class IDF1Metrics:
    idf1: float
    idp: float  # ID Precision
    idr: float  # ID Recall
    idtp: int  # ID True Positives
    idfp: int  # ID False Positives
    idfn: int  # ID False Negatives


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def match_detections(
    detections: List[np.ndarray], ground_truths: List[np.ndarray], iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Empareja detecciones con ground truths usando IoU máximo con un umbral mínimo de 0.5.
    Retorna las parejas coincidentes y los índices no emparejados.
    """
    if len(detections) == 0 or len(ground_truths) == 0:
        print("No hay detecciones o ground truths disponibles.")

        return [], list(range(len(detections))), list(range(len(ground_truths)))

    matches = []
    matched_gts = set()
    unmatched_dets = list(range(len(detections)))
    unmatched_gts = list(range(len(ground_truths)))

    print(f"Detecciones: {len(detections)}, Ground truths: {len(ground_truths)}")
    print(f"Detecciones: {detections}")
    print(f"Ground truths: {ground_truths}")

    # Para cada detección, encuentra el ground truth con el mayor IoU
    for i, det in enumerate(detections):
        max_iou = iou_threshold
        best_match = -1

        for j, gt in enumerate(ground_truths):
            if j in matched_gts:
                continue

            try:
                iou = calculate_iou(det, gt)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
            except Exception as e:
                print(f"Error calculando IoU: {e}")
                print(f"Detección: {det}")
                print(f"Ground truth: {gt}")

        # Si encontramos una coincidencia válida
        if best_match != -1:
            print(
                f"Coincidencia encontrada: Detección {i} con GT {best_match} (IoU: {max_iou:.2f})"
            )
            matches.append((i, best_match))
            matched_gts.add(best_match)
            unmatched_dets.remove(i)
            unmatched_gts.remove(best_match)

    if len(matches) == 0:
        print("No se encontraron coincidencias.")
    else:
        print(f"Se encontraron {len(matches)} coincidencias.")

    return matches, unmatched_dets, unmatched_gts


class TrackingMetrics:
    def __init__(self):
        self.total_idtp = 0  # Total ID True Positives
        self.total_idfp = 0  # Total ID False Positives
        self.total_idfn = 0  # Total ID False Negatives
        self.track_history = {}  # Historial de tracks {track_id: {frame_id: ground_truth_id}}

    def update(
        self,
        frame_id: int,
        detections: List[Tuple[int, np.ndarray]],
        ground_truths: List[Tuple[int, np.ndarray]],
        iou_threshold: float = 0.5,
    ):
        """
        Actualiza las métricas con las detecciones del frame actual.

        Args:
            frame_id: ID del frame actual
            detections: Lista de tuplas (track_id, bounding_box)
            ground_truths: Lista de tuplas (gt_id, bounding_box)
            iou_threshold: Umbral de IoU para considerar una coincidencia
        """
        det_boxes = [d[1] for d in detections]
        gt_boxes = [g[1] for g in ground_truths]

        matches, unmatched_dets, unmatched_gts = match_detections(
            det_boxes, gt_boxes, iou_threshold
        )

        # Actualizar True Positives
        for det_idx, gt_idx in matches:
            det_track_id = detections[det_idx][0]
            gt_id = ground_truths[gt_idx][0]

            if det_track_id not in self.track_history:
                self.track_history[det_track_id] = {}

            # Si el track mantiene consistencia con el mismo ground truth
            if all(gt == gt_id for gt in self.track_history[det_track_id].values()):
                self.total_idtp += 1
            else:
                self.total_idfp += 1

            self.track_history[det_track_id][frame_id] = gt_id

        # Actualizar False Positives
        self.total_idfp += len(unmatched_dets)

        # Actualizar False Negatives
        self.total_idfn += len(unmatched_gts)

    def compute(self) -> IDF1Metrics:
        """Calcula las métricas IDF1 finales."""
        idtp = self.total_idtp
        idfp = self.total_idfp
        idfn = self.total_idfn

        idp = idtp / (idtp + idfp) if idtp + idfp > 0 else 0
        idr = idtp / (idtp + idfn) if idtp + idfn > 0 else 0
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn > 0 else 0

        return IDF1Metrics(idf1, idp, idr, idtp, idfp, idfn)
