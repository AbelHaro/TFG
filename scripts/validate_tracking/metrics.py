import numpy as np
from typing import List, Tuple
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
    total_iou: float  # Sum of IoU for all correct matches
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
    return intersection / union if union > 0 else 0.0


def match_detections(
    detections: List[np.ndarray], ground_truths: List[np.ndarray], iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Empareja detecciones con ground truths usando el algoritmo Húngaro basado en IoU.
    Retorna las parejas coincidentes y los índices no emparejados.
    """
    if len(detections) == 0 or len(ground_truths) == 0:
        return [], list(range(len(detections))), list(range(len(ground_truths)))

    cost_matrix = np.full((len(detections), len(ground_truths)), 1000.0)
    valid = False
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            iou = calculate_iou(det, gt)
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1 - iou
                valid = True
    if not valid:
        return [], list(range(len(detections))), list(range(len(ground_truths)))

    from scipy.optimize import linear_sum_assignment

    det_idx, gt_idx = linear_sum_assignment(cost_matrix)

    matches, unmatched_dets, unmatched_gts = (
        [],
        list(range(len(detections))),
        list(range(len(ground_truths))),
    )
    for d, g in zip(det_idx, gt_idx):
        if cost_matrix[d, g] < 1000.0:
            matches.append((d, g))
            unmatched_dets.remove(d)
            unmatched_gts.remove(g)
    return matches, unmatched_dets, unmatched_gts


class TrackingMetrics:
    def __init__(self):
        # IDF1 metrics
        self.total_idtp = 0
        self.total_idfp = 0
        self.total_idfn = 0
        self.track_history = {}

        # HOTA metrics
        self.alpha_range = np.arange(0.05, 1.0, 0.05)
        self.hota_tps = {a: 0 for a in self.alpha_range}
        self.hota_fps = {a: 0 for a in self.alpha_range}
        self.hota_fns = {a: 0 for a in self.alpha_range}
        self.assoc_scores = {a: [] for a in self.alpha_range}

        # MOTA metrics
        self.mota_fp = 0
        self.mota_fn = 0
        self.mota_idsw = 0
        self.gt_total = 0
        self.prev_matches = {}

        # MOTP metrics (acumular IoU)
        self.total_iou = 0.0
        self.total_matches = 0

    def update(
        self,
        frame_id: int,
        detections: List[Tuple[int, np.ndarray]],
        ground_truths: List[Tuple[int, np.ndarray]],
        iou_threshold: float = 0.5,
    ):
        det_boxes = [d[1] for d in detections]
        gt_boxes = [g[1] for g in ground_truths]

        # MOTA ground truth count
        self.gt_total += len(ground_truths)

        # Matching
        matches, u_dets, u_gts = match_detections(det_boxes, gt_boxes, iou_threshold)

        # Update IDF1 and MOTP
        for d_idx, g_idx in matches:
            track_id = detections[d_idx][0]
            gt_id = ground_truths[g_idx][0]

            iou = calculate_iou(det_boxes[d_idx], gt_boxes[g_idx])
            self.total_iou += iou
            self.total_matches += 1

            if track_id not in self.track_history:
                self.track_history[track_id] = {}

            if all(prev == gt_id for prev in self.track_history[track_id].values()):
                self.total_idtp += 1
            else:
                self.total_idfp += 1
            self.track_history[track_id][frame_id] = gt_id

        self.total_idfp += len(u_dets)
        self.total_idfn += len(u_gts)

        # Update MOTA FP/FN
        self.mota_fp += len(u_dets)
        self.mota_fn += len(u_gts)

        # Identity switches
        curr_matches = {}
        for d_idx, g_idx in matches:
            tid = detections[d_idx][0]
            gid = ground_truths[g_idx][0]
            curr_matches[gid] = tid
            if gid in self.prev_matches and self.prev_matches[gid] != tid:
                self.mota_idsw += 1
        self.prev_matches = curr_matches

        # Update HOTA
        for a in self.alpha_range:
            alpha_matches = [
                (i, j, calculate_iou(det_boxes[i], gt_boxes[j]))
                for i in range(len(det_boxes))
                for j in range(len(gt_boxes))
                if calculate_iou(det_boxes[i], gt_boxes[j]) >= a
            ]
            if alpha_matches:
                alpha_matches.sort(key=lambda x: x[2], reverse=True)
                md, mg = set(), set()
                for i, j, sc in alpha_matches:
                    if i not in md and j not in mg:
                        md.add(i)
                        mg.add(j)
                        self.assoc_scores[a].append(sc)
                self.hota_tps[a] += len(md)
                self.hota_fps[a] += len(det_boxes) - len(md)
                self.hota_fns[a] += len(gt_boxes) - len(mg)
            else:
                self.hota_fps[a] += len(det_boxes)
                self.hota_fns[a] += len(gt_boxes)

    def compute(self) -> Tuple[IDF1Metrics, HOTAMetrics, MOTAMetrics, MOTPMetrics]:
        # IDF1
        idtp, idfp, idfn = self.total_idtp, self.total_idfp, self.total_idfn
        idp = idtp / (idtp + idfp) if idtp + idfp > 0 else 0
        idr = idtp / (idtp + idfn) if idtp + idfn > 0 else 0
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn > 0 else 0

        # HOTA
        hota_scores, deta_scores, assa_scores = [], [], []
        for a in self.alpha_range:
            tp, fp, fn = self.hota_tps[a], self.hota_fps[a], self.hota_fns[a]
            deta = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
            assa = np.mean(self.assoc_scores[a]) if self.assoc_scores[a] else 0
            hota_scores.append(np.sqrt(deta * assa) if deta > 0 and assa > 0 else 0)
            deta_scores.append(deta)
            assa_scores.append(assa)
        final_hota = np.mean(hota_scores)
        final_deta = np.mean(deta_scores)
        final_assa = np.mean(assa_scores)
        mid_tp = self.hota_tps[0.5]
        mid_fp = self.hota_fps[0.5]
        mid_fn = self.hota_fns[0.5]

        # MOTA
        mota = 1 - (self.mota_fn + self.mota_fp + self.mota_idsw) / max(self.gt_total, 1)

        # MOTP usando IoU
        motp = self.total_iou / max(self.total_matches, 1)

        return (
            IDF1Metrics(idf1, idp, idr, idtp, idfp, idfn),
            HOTAMetrics(final_hota, final_deta, final_assa, mid_tp, mid_fp, mid_fn),
            MOTAMetrics(mota, self.mota_fp, self.mota_fn, self.mota_idsw, self.gt_total),
            MOTPMetrics(motp, self.total_iou, self.total_matches),
        )
