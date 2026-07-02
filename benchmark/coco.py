from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from functools import cache

import geopandas as gpd
import numpy as np
import rasterio
from pycocotools import mask as mask_utils

try:
    # faster-coco-eval reimplements evaluateImg/accumulate in C++ (~30x faster
    # than pycocotools) with identical metrics; fall back to pycocotools if unavailable.
    import logging

    from faster_coco_eval import COCO
    from faster_coco_eval import COCOeval_faster as COCOeval

    # Set logging level to WARNING to suppress info messages from faster_coco_eval
    logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

from shapely.geometry import MultiPolygon, Polygon

from spm import SPMPrediction


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

def _polygon_to_rle(polygon: Polygon | MultiPolygon, height: int, width: int):
    parts = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
    coords = [np.asarray(p.exterior.coords).ravel().tolist()
              for p in parts if not p.is_empty and len(p.exterior.coords) >= 3]
    if not coords:
        return None
    rles = mask_utils.frPyObjects(coords, height, width)
    return mask_utils.merge(rles)

@cache
def _ground_truth_polygons(label_path: Path, image_path: Path) -> list[Polygon]:
    """Load GPKG labels and project them into the crop's pixel coordinates."""
    with rasterio.open(image_path) as src:
        inverse = ~src.transform

    def to_pixels(x, y, z=None):
        return inverse * (x, y)

    from shapely.ops import transform

    labels = gpd.read_file(label_path)
    polygons: list[Polygon] = []
    for geom in labels.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            polygons.append(transform(to_pixels, part))
    return polygons


def _rle_entry(rle):
    counts = rle["counts"]
    return {"size": rle["size"], "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def evaluate_prediction(
    prediction: SPMPrediction,
    label_path: str | Path,
    iou_threshold: float = 0.5,
) -> MergeMetrics:
    """Score a merged prediction against ground-truth labels with pycocotools.

    mAP/mAP50/mAP75 come from COCOeval (segm). Precision, recall and F1 are computed
    at ``iou_threshold`` by greedy score-ordered matching on the same RLE masks.
    """
    height, width = prediction.image_shape
    gt_polygons = _ground_truth_polygons(Path(label_path), Path(prediction.image_path))

    gt_rles = [r for r in (_polygon_to_rle(p, height, width) for p in gt_polygons) if r]

    order = np.argsort(prediction.confidences)[::-1] if prediction.count else []
    dt_rles, dt_scores = [], []
    for i in order:
        rle = _polygon_to_rle(prediction.polygons[i], height, width)
        if rle is None:
            continue
        dt_rles.append(rle)
        dt_scores.append(float(prediction.confidences[i]))

    precision, recall, f1 = _prf(dt_rles, gt_rles, iou_threshold)
    mAP, mAP50, mAP75 = _coco_map(dt_rles, dt_scores, gt_rles, height, width)

    return MergeMetrics(
        num_gt=len(gt_rles), num_pred=len(dt_rles),
        mAP=mAP, mAP50=mAP50, mAP75=mAP75,
        precision=precision, recall=recall, f1=f1,
    )


def _prf(dt_rles, gt_rles, iou_threshold) -> tuple[float, float, float]:
    if not gt_rles and not dt_rles:
        return 1.0, 1.0, 1.0
    if not dt_rles:
        return 0.0, 0.0, 0.0
    if not gt_rles:
        return 0.0, 0.0, 0.0

    ious = mask_utils.iou(dt_rles, gt_rles, [0] * len(gt_rles))
    matched_gt: set[int] = set()
    tp = 0
    for d in range(len(dt_rles)):
        best_iou, best_g = iou_threshold, -1
        for g in range(len(gt_rles)):
            if g in matched_gt:
                continue
            if ious[d, g] >= best_iou:
                best_iou, best_g = ious[d, g], g
        if best_g >= 0:
            matched_gt.add(best_g)
            tp += 1

    fp = len(dt_rles) - tp
    fn = len(gt_rles) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _coco_map(dt_rles, dt_scores, gt_rles, height, width) -> tuple[float, float, float]:
    if not gt_rles:
        return 0.0, 0.0, 0.0

    gt = {
        "images": [{"id": 1, "height": height, "width": width}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "segmentation": _rle_entry(r), "area": float(mask_utils.area(r)),
             "bbox": mask_utils.toBbox(r).tolist()}
            for i, r in enumerate(gt_rles)
        ],
    }
    detections = [
        {"image_id": 1, "category_id": 1, "score": dt_scores[i],
         "segmentation": _rle_entry(r), "bbox": mask_utils.toBbox(r).tolist()}
        for i, r in enumerate(dt_rles)
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections) if detections else COCO()

        if not detections:
            return 0.0, 0.0, 0.0

        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        # Default maxDets caps evaluation at the top-100 detections, which
        # underestimates mAP on dense crops with hundreds of objects.
        coco_eval.params.maxDets = [1, 10, max(len(dt_rles), 100)]
        coco_eval.params.areaRng = [coco_eval.params.areaRng[0]]   # keep only "all"
        coco_eval.params.areaRngLbl = ["all"]
        coco_eval.evaluate()
        coco_eval.accumulate()

    # Read AP directly from the precision array rather than summarize(), whose
    # mAP line hardcodes a maxDets=100 lookup that breaks once we raise the cap.
    # precision shape: [iouThr, recThr, cat, areaRng, maxDet]; index 0 of
    # areaRng = "all", index -1 of maxDet = our raised cap.
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]

    def _mean_ap(p):
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else 0.0

    mAP = _mean_ap(precision)            # IoU 0.50:0.95
    mAP50 = _mean_ap(precision[0])       # IoU 0.50
    mAP75 = _mean_ap(precision[5])       # IoU 0.75
    return mAP, mAP50, mAP75
