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


@dataclass
class HOTAMetrics:
    hota: float
    deta: float  # Detection Accuracy
    assa: float  # Association Accuracy
    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives


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
        # IDF1 metrics
        self.total_idtp = 0  # Total ID True Positives
        self.total_idfp = 0  # Total ID False Positives
        self.total_idfn = 0  # Total ID False Negatives
        self.track_history = {}  # Historial de tracks {track_id: {frame_id: ground_truth_id}}

        # HOTA metrics
        self.alpha_range = np.arange(0.05, 1.0, 0.05)  # 19 valores de α de 0.05 a 0.95
        self.hota_tps = {alpha: 0 for alpha in self.alpha_range}
        self.hota_fps = {alpha: 0 for alpha in self.alpha_range}
        self.hota_fns = {alpha: 0 for alpha in self.alpha_range}
        self.association_scores = {alpha: [] for alpha in self.alpha_range}

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

        # Actualizar métricas IDF1
        matches, unmatched_dets, unmatched_gts = match_detections(
            det_boxes, gt_boxes, iou_threshold
        )

        # Actualizar True Positives para IDF1
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

        # Actualizar False Positives y Negatives para IDF1
        self.total_idfp += len(unmatched_dets)
        self.total_idfn += len(unmatched_gts)

        # Actualizar métricas HOTA para cada umbral alpha
        for alpha in self.alpha_range:
            # Para cada alpha, realizar matching usando ese umbral
            alpha_matches = []
            for i, det in enumerate(det_boxes):
                for j, gt in enumerate(gt_boxes):
                    iou = calculate_iou(det, gt)
                    if iou >= alpha:
                        alpha_matches.append((i, j, iou))

            if alpha_matches:
                # Ordenar matches por IoU descendente
                alpha_matches.sort(key=lambda x: x[2], reverse=True)
                matched_dets = set()
                matched_gts = set()
                final_matches = []

                # Greedy matching basado en IoU
                for det_idx, gt_idx, iou_score in alpha_matches:
                    if det_idx not in matched_dets and gt_idx not in matched_gts:
                        final_matches.append((det_idx, gt_idx, iou_score))
                        matched_dets.add(det_idx)
                        matched_gts.add(gt_idx)

                # Actualizar conteos para este alpha
                self.hota_tps[alpha] += len(final_matches)
                self.hota_fps[alpha] += len(det_boxes) - len(matched_dets)
                self.hota_fns[alpha] += len(gt_boxes) - len(matched_gts)

                # Actualizar scores de asociación
                for _, _, iou in final_matches:
                    self.association_scores[alpha].append(iou)
            else:
                self.hota_fps[alpha] += len(det_boxes)
                self.hota_fns[alpha] += len(gt_boxes)

    def compute(self) -> Tuple[IDF1Metrics, HOTAMetrics]:
        """Calcula las métricas IDF1 y HOTA finales."""
        # Calcular IDF1
        idtp = self.total_idtp
        idfp = self.total_idfp
        idfn = self.total_idfn

        idp = idtp / (idtp + idfp) if idtp + idfp > 0 else 0
        idr = idtp / (idtp + idfn) if idtp + idfn > 0 else 0
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn > 0 else 0

        # Calcular HOTA
        hota_scores = []
        deta_scores = []
        assa_scores = []

        for alpha in self.alpha_range:
            tp = self.hota_tps[alpha]
            fp = self.hota_fps[alpha]
            fn = self.hota_fns[alpha]

            # Detection Accuracy (DetA)
            deta = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            deta_scores.append(deta)

            # Association Accuracy (AssA)
            if tp > 0 and len(self.association_scores[alpha]) > 0:
                assa = np.mean(self.association_scores[alpha])
            else:
                assa = 0
            assa_scores.append(assa)

            # HOTA score para este alpha
            hota = np.sqrt(deta * assa) if deta > 0 and assa > 0 else 0
            hota_scores.append(hota)

        # Promediar los scores sobre todos los alphas
        final_hota = np.mean(hota_scores)
        final_deta = np.mean(deta_scores)
        final_assa = np.mean(assa_scores)

        # Usar los valores del alpha medio (0.5) para los conteos
        mid_alpha = 0.5
        final_tp = self.hota_tps[mid_alpha]
        final_fp = self.hota_fps[mid_alpha]
        final_fn = self.hota_fns[mid_alpha]

        return (
            IDF1Metrics(idf1, idp, idr, idtp, idfp, idfn),
            HOTAMetrics(final_hota, final_deta, final_assa, final_tp, final_fp, final_fn),
        )
