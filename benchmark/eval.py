from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import cache

import geopandas as gpd
import numpy as np
import rasterio
from shapely import STRtree, make_valid
from shapely.geometry import base as shapely_base

from spm import SPMPrediction

# IoU sweep: 0.50, 0.55, ..., 0.95. Index 0 -> mAP50, index 5 -> mAP75.
_IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)
_REC_THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass
class MergeMetrics:
    """Detection scores for one merged prediction against its ground truth."""

    num_gt: int
    num_pred: int
    mAP: float
    mAP50: float
    mAP75: float
    precision: float
    recall: float
    f1: float


def _clean(geom) -> shapely_base.BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    return None if geom.is_empty or geom.area == 0 else geom


@cache
def _ground_truth_polygons(label_path: Path, image_path: Path) -> list:
    """Load GPKG labels and project them into the crop's pixel coordinates."""
    with rasterio.open(image_path) as src:
        inverse = ~src.transform

    def to_pixels(x, y, z=None):
        return inverse * (x, y)

    from shapely.ops import transform

    labels = gpd.read_file(label_path)
    polygons = []
    for geom in labels.geometry:
        cleaned = _clean(geom)
        if cleaned is None:
            continue
        parts = cleaned.geoms if cleaned.geom_type == "MultiPolygon" else [cleaned]
        for part in parts:
            polygons.append(transform(to_pixels, part))
    return polygons


def _ground_truth_dict(label_path: Path, image_path: Path) -> dict:
    """Load GPKG labels and project them into the crop's pixel coordinates.

    Returns an entry dict:
    ``image_name``, ``image_size`` (H, W) and an ``annotations`` list whose
    ``segmentation`` fields are nested ``[[x, y], ...]`` pixel rings.
    """
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width

    annotations: list[dict] = []
    for polygon in _ground_truth_polygons(Path(label_path), Path(image_path)):
        coords = [
            [int(round(px)), int(round(py))] for px, py in polygon.exterior.coords
        ]
        if len(coords) < 3:
            continue
        minx, miny, maxx, maxy = polygon.bounds
        bbox = [int(round(minx)), int(round(miny)), int(round(maxx)), int(round(maxy))]
        annotations.append(
            {
                "type": "object",
                "class_id": 0,
                "bbox": bbox,
                "area": float(polygon.area),
                "segmentation": [coords],
                "confidence": 1.0,
            }
        )

    return {
        "image_name": Path(image_path).name,
        "image_size": (height, width),
        "annotations": annotations,
    }


def _candidate_ious(dt_polygons, gt_polygons) -> list[list[tuple[float, int]]]:
    """For each detection, the (iou, gt_index) pairs of GTs whose envelopes overlap.

    An STRtree over the GT polygons restricts the exact geometric IoU to
    spatially-overlapping candidates, so cost scales with local density rather
    than the full N_dt x N_gt grid. IoU is computed directly on the polygon
    geometries (intersection area / union area) — no rasterization.
    """
    if not gt_polygons:
        return [[] for _ in dt_polygons]
    tree = STRtree(gt_polygons)
    candidates: list[list[tuple[float, int]]] = []
    for dt in dt_polygons:
        pairs: list[tuple[float, int]] = []
        for g in tree.query(dt):  # envelope-intersection candidates
            gt = gt_polygons[int(g)]
            inter = dt.intersection(gt).area
            if inter <= 0.0:
                continue
            union = dt.area + gt.area - inter
            if union > 0.0:
                pairs.append((inter / union, int(g)))
        candidates.append(pairs)
    return candidates


def _prf(candidates, num_gt, iou_threshold) -> tuple[float, float, float]:
    """Greedy score-ordered matching at a single IoU threshold."""
    if num_gt == 0 and not candidates:
        return 1.0, 1.0, 1.0
    if not candidates or num_gt == 0:
        return 0.0, 0.0, 0.0

    matched: set[int] = set()
    tp = 0
    for pairs in candidates:  # candidates are in descending-score order
        best_iou, best_g = iou_threshold, -1
        for iou, g in pairs:
            if g in matched:
                continue
            if iou >= best_iou:
                best_iou, best_g = iou, g
        if best_g >= 0:
            matched.add(best_g)
            tp += 1

    fp = len(candidates) - tp
    fn = num_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _average_precision(candidates, num_gt) -> tuple[float, float, float]:
    """COCO-style AP over the IoU sweep, computed from the pruned candidate pairs.

    Detections are assumed pre-sorted by descending score. Returns (mAP, mAP50, mAP75).
    """
    if num_gt == 0 or not candidates:
        return 0.0, 0.0, 0.0

    n = len(candidates)
    aps = np.zeros(len(_IOU_THRESHOLDS))
    for t_idx, t in enumerate(_IOU_THRESHOLDS):
        tp = np.zeros(n)
        fp = np.zeros(n)
        matched: set[int] = set()
        for d, pairs in enumerate(candidates):
            best_iou, best_g = t, -1
            for iou, g in pairs:
                if g in matched:
                    continue
                if iou >= best_iou:
                    best_iou, best_g = iou, g
            if best_g >= 0:
                matched.add(best_g)
                tp[d] = 1
            else:
                fp[d] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # Monotonic precision envelope from the right, as in COCO's accumulate.
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        idx = np.searchsorted(recall, _REC_THRESHOLDS, side="left")
        interp = np.zeros(len(_REC_THRESHOLDS))
        valid = idx < n
        interp[valid] = precision[idx[valid]]
        aps[t_idx] = interp.mean()

    return float(aps.mean()), float(aps[0]), float(aps[5])


def evaluate_prediction(
    prediction: SPMPrediction,
    label_path: str | Path,
    iou_threshold: float = 0.5,
) -> MergeMetrics:
    """Score a merged prediction against ground-truth labels using polygon IoU.

    mAP/mAP50/mAP75 follow the COCO AP protocol; precision/recall/F1 use greedy
    score-ordered matching at ``iou_threshold``. IoU is computed directly on the
    polygon geometries, with candidate pairs pre-filtered by an STRtree over the
    ground-truth polygons so exact IoU is only evaluated for spatially-overlapping
    objects.
    """
    if prediction.image_path is None:
        raise ValueError("prediction.image_path is not set.")
    gt_polygons = _ground_truth_polygons(Path(label_path), Path(prediction.image_path))

    order = np.argsort(prediction.confidences)[::-1] if prediction.count else []
    dt_polygons = []
    for i in order:
        cleaned = _clean(prediction.polygons[i])
        if cleaned is not None:
            dt_polygons.append(cleaned)

    candidates = _candidate_ious(dt_polygons, gt_polygons)
    precision, recall, f1 = _prf(candidates, len(gt_polygons), iou_threshold)
    mAP, mAP50, mAP75 = _average_precision(candidates, len(gt_polygons))

    return MergeMetrics(
        num_gt=len(gt_polygons),
        num_pred=len(dt_polygons),
        mAP=mAP,
        mAP50=mAP50,
        mAP75=mAP75,
        precision=precision,
        recall=recall,
        f1=f1,
    )
