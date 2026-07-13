from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from benchmark.coco import MergeMetrics, evaluate_prediction
from benchmark.methods import MERGING_METHODS, MergingMethod
from spm.config.config import ModelConfig
from spm.core.prediction import SPMPrediction
from spm.models.yolo import YOLOModel
from spm.utils.logger import logger


def _unmerged_prediction(model: YOLOModel, image_path: Path) -> SPMPrediction:
    """Run tiled inference once per crop; merging is applied separately per method."""
    unmerged = model.predict(image_path)
    return unmerged


def benchmark_test_set(
    test_set_dir: str | Path,
    model_config: ModelConfig,
    methods: list[MergingMethod] | None = None,
    iou_threshold: float = 0.5,
    output_csv: str | Path | None = None,
    crop_limit: int | None = None,
) -> list[dict]:
    """Evaluate every merging method on every crop of a generated test set.

    Walks ``test_set_dir`` for ``crop_*/image.tif`` + ``labels.gpkg`` pairs, runs the
    YOLO tiled inference once per crop, applies each merging method to that shared set
    of raw detections, and scores the result with pycocotools.

    Returns one row per (crop, method); writes the same rows to ``output_csv`` if given.
    """
    test_set_dir = Path(test_set_dir)
    methods = methods or list(MERGING_METHODS.values())
    model = YOLOModel(model_config)

    rows: list[dict] = []
    crops = sorted(test_set_dir.glob("crop_*"))
    if crop_limit is not None:
        crops = crops[:crop_limit]
    logger.info(f"Benchmarking {len(crops)} crops with {len(methods)} methods")

    for crop_dir in crops:
        image_path = crop_dir / "image.tif"
        label_path = crop_dir / "labels.gpkg"
        if not (image_path.exists() and label_path.exists()):
            continue

        crop_size = int(crop_dir.name.split("_")[1])
        unmerged = _unmerged_prediction(model, image_path)

        for method in methods:
            try:
                pred_output_dir = Path("benchmark_output") / crop_dir.name / method.name
                merged, merge_time, peak_memory_usage = method.merge(unmerged)
                merged.save_to_file(output_dir=pred_output_dir)
                metrics = evaluate_prediction(merged, label_path, iou_threshold)
            except Exception as exc:
                logger.error(f"{method.name} failed on {crop_dir.name}: {exc}")
                continue

            rows.append({
                "crop": crop_dir.name,
                "crop_size": crop_size,
                "method": method.name,
                "merge_time": merge_time,
                "peak_memory_usage": peak_memory_usage,
                **asdict(metrics),
            })
            logger.info(
                f"{crop_dir.name} / {method.name}: "
                f"mAP={metrics.mAP:.3f} P={metrics.precision:.3f} "
                f"R={metrics.recall:.3f} F1={metrics.f1:.3f} "
                f"merge_time={merge_time:.3f}s peak_memory_usage={peak_memory_usage:.3f}MiB"
                )

    if output_csv and rows:
        _write_csv(rows, Path(output_csv))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Average each metric per (crop_size, method) across crops."""
    metric_keys = [k for k in MergeMetrics.__annotations__ if k not in ("num_gt", "num_pred")]
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["crop_size"], row["method"]), []).append(row)

    summary = []
    for (crop_size, method), group in sorted(groups.items()):
        entry = {"crop_size": crop_size, "method": method, "num_crops": len(group)}
        for key in metric_keys:
            entry[key] = sum(r[key] for r in group) / len(group)
        summary.append(entry)
    return summary


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {path}")
