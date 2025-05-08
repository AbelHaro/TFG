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


@dataclass
class MOTAMetrics:
    mota: float
    fp: int
    fn: int
    idsw: int  # Identity Switches
    gt_total: int  # Total ground truth objects


@dataclass
class MOTPMetrics:
    motp: float  # Multiple Object Tracking Precision
    total_distance: float  # Sum of distances for all correct matches
    total_matches: int  # Total number of correct matches


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
    Empareja detecciones con ground truths usando el algoritmo Húngaro basado en IoU.
    Retorna las parejas coincidentes y los índices no emparejados.
    """
    if len(detections) == 0 or len(ground_truths) == 0:
        print("No hay detecciones o ground truths disponibles.")
        return [], list(range(len(detections))), list(range(len(ground_truths)))

    # Crear matriz de costos basada en IoU
    cost_matrix = np.full((len(detections), len(ground_truths)), 1000.0)  # Alto costo por defecto
    valid_assignments = False

    # Calcular matriz de IoU y actualizar costos
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            try:
                iou = calculate_iou(det, gt)
                if iou >= iou_threshold:
                    cost_matrix[i, j] = 1 - iou  # Costo bajo para buenos matches
                    valid_assignments = True
            except Exception as e:
                print(f"Error calculando IoU: {e}")
                print(f"Detección: {det}")
                print(f"Ground truth: {gt}")

    if not valid_assignments:
        return [], list(range(len(detections))), list(range(len(ground_truths)))

    from scipy.optimize import linear_sum_assignment

    try:
        # Aplicar algoritmo Húngaro
        det_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_gts = list(range(len(ground_truths)))

        # Filtrar coincidencias válidas (costo < 1000.0)
        for det_idx, gt_idx in zip(det_indices, gt_indices):
            if cost_matrix[det_idx, gt_idx] < 1000.0:  # Solo incluir matches válidos
                matches.append((det_idx, gt_idx))
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)
                if gt_idx in unmatched_gts:
                    unmatched_gts.remove(gt_idx)

        print(f"Se encontraron {len(matches)} coincidencias usando el algoritmo Húngaro.")
        return matches, unmatched_dets, unmatched_gts

    except Exception as e:
        print(f"Error en el algoritmo Húngaro: {e}")
        return [], list(range(len(detections))), list(range(len(ground_truths)))


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

        # MOTA metrics
        self.mota_fp = 0  # Total False Positives for MOTA
        self.mota_fn = 0  # Total False Negatives for MOTA
        self.mota_idsw = 0  # Total Identity Switches
        self.gt_total = 0  # Total ground truth objects
        self.prev_matches = {}  # Previous frame matches {gt_id: track_id}

        # MOTP metrics
        self.total_distance = 0.0  # Sum of distances for all correct matches
        self.total_matches = 0  # Total number of correct matches

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

        # Actualizar total de ground truths para MOTA
        self.gt_total += len(ground_truths)

        # Actualizar métricas IDF1 y MOTP
        matches, unmatched_dets, unmatched_gts = match_detections(
            det_boxes, gt_boxes, iou_threshold
        )

        # Actualizar True Positives para IDF1 y MOTP
        for det_idx, gt_idx in matches:
            det_track_id = detections[det_idx][0]
            gt_id = ground_truths[gt_idx][0]

            # Calcular distancia para MOTP (1 - IoU)
            iou = calculate_iou(det_boxes[det_idx], gt_boxes[gt_idx])
            distance = 1 - iou
            self.total_distance += distance
            self.total_matches += 1

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

        # Actualizar métricas MOTA
        current_matches = {}

        # Actualizar False Positives y False Negatives para MOTA
        self.mota_fp += len(unmatched_dets)  # Detecciones sin match son FP
        self.mota_fn += len(unmatched_gts)  # Ground truths sin match son FN

        # Detectar cambios de identidad para MOTA
        for det_idx, gt_idx in matches:
            det_track_id = detections[det_idx][0]
            gt_id = ground_truths[gt_idx][0]
            current_matches[gt_id] = det_track_id

            # Verificar si hubo un cambio de identidad con respecto al frame anterior
            if gt_id in self.prev_matches and self.prev_matches[gt_id] != det_track_id:
                self.mota_idsw += 1

        # Actualizar matches previos para el siguiente frame
        self.prev_matches = current_matches

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

    def compute(self) -> Tuple[IDF1Metrics, HOTAMetrics, MOTAMetrics, MOTPMetrics]:
        """Calcula las métricas IDF1, HOTA, MOTA y MOTP finales."""
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

        # Calcular MOTA
        mota = 1 - (self.mota_fn + self.mota_fp + self.mota_idsw) / max(self.gt_total, 1)

        # Calcular MOTP
        motp = self.total_distance / max(self.total_matches, 1)

        return (
            IDF1Metrics(idf1, idp, idr, idtp, idfp, idfn),
            HOTAMetrics(final_hota, final_deta, final_assa, final_tp, final_fp, final_fn),
            MOTAMetrics(mota, self.mota_fp, self.mota_fn, self.mota_idsw, self.gt_total),
            MOTPMetrics(motp, self.total_distance, self.total_matches),
        )
