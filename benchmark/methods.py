from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np
import psutil

from spm.config.config import SPMConfig
from spm.core.prediction import SPMPrediction
from spm.merging.spm import SpatialPolygonMerger
from spm.preprocessing.image_processing import binary_mask_to_contours

MergeFn = Callable[..., SPMPrediction]

# RSS poll interval for the in-child memory monitor. Small enough to catch
# short native allocation spikes that a coarse sampler would miss.
_RSS_POLL_INTERVAL = 0.001


def _sample_peak_rss(proc: psutil.Process, stop: threading.Event, out: list) -> None:
    """Track the high-water RSS of ``proc`` until ``stop`` is set."""
    peak = proc.memory_info().rss
    while not stop.is_set():
        try:
            peak = max(peak, proc.memory_info().rss)
        except psutil.Error:
            break
    out[0] = peak


def _merge_worker(fn: MergeFn, unmerged: SPMPrediction, args: tuple,
                  kwargs: dict, queue: mp.Queue) -> None:
    """Run one merge in an isolated process and report merged result + memory.

    Peak RSS is sampled around the call so the reported figure includes
    C-extension / native allocations (OpenCV, LSNMS, numpy), which a
    Python-only profiler such as tracemalloc cannot see. Running in a fresh
    child means each method starts from identical state, so the peak-minus-
    baseline delta isn't polluted by memory another method left resident.

    The child is spawned (not forked) because the parent has already
    initialized CUDA for inference, and a CUDA context cannot survive a fork.
    CUDA is then disabled here so the merge runs on CPU: these are
    postprocessing ops, and a per-child CUDA context would otherwise dump
    hundreds of MiB of host memory into the measurement. This must run before
    torch is first imported (lazily, inside the merge functions).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    proc = psutil.Process()
    peak_holder = [0]
    stop = threading.Event()
    monitor = threading.Thread(target=_sample_peak_rss, args=(proc, stop, peak_holder))

    baseline = proc.memory_info().rss
    monitor.start()
    try:
        start_time = time.perf_counter()
        merged = fn(unmerged, *args, **kwargs)
        time_elapsed = time.perf_counter() - start_time
    finally:
        stop.set()
        monitor.join()

    peak_bytes = max(peak_holder[0] - baseline, 0)
    queue.put((merged, time_elapsed, peak_bytes / (1024 * 1024)))


class MergingMethod:
    """Names a merging strategy and applies it to an unmerged prediction.

    Every method takes the raw, tile-level detections (one SPMPrediction in crop
    pixel coordinates) and returns a merged SPMPrediction in the same coordinate
    space, so results are directly comparable against the ground truth.
    """

    def __init__(self, name: str, fn: MergeFn,  *args, **kwargs):
        self.name = name
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def merge(self, unmerged: SPMPrediction) -> tuple[SPMPrediction, float, float]:
        # Isolate each merge in a child process so the peak-RSS measurement
        # (which captures C-extension memory) starts from identical state and
        # isn't affected by allocations left resident by previous methods.
        # "spawn" rather than "fork" because the parent has CUDA initialized
        # for inference, which a fork cannot inherit safely.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        worker = ctx.Process(
            target=_merge_worker,
            args=(self._fn, unmerged, self._args, self._kwargs, queue),
        )
        worker.start()
        merged, time_elapsed, peak_memory_usage = queue.get()
        worker.join()

        merged.image_path = unmerged.image_path
        merged.image_shape = unmerged.image_shape
        return merged, time_elapsed, peak_memory_usage


def _frame_shape(prediction: SPMPrediction) -> tuple[int, int]:
    if prediction.image_shape is None:
        raise ValueError("prediction.image_shape (height, width) is required for merging.")
    return prediction.image_shape


def _rasterize(polygon, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    parts = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    for part in parts:
        if part.is_empty:
            continue
        pts = np.asarray(part.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    return mask


def _add_mask_annotation(prediction: SPMPrediction, mask: np.ndarray, name, class_id, confidence) -> None:
    mask = np.ascontiguousarray(mask).astype(np.uint8)
    contours = binary_mask_to_contours(mask, min_area=1, normalize=False, sort_by_area=True)
    if not contours:
        return
    segmentation = [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in contours]
    prediction.add_annotation(
        name=name, class_id=int(class_id), confidence=float(confidence),
        segmentation=segmentation)


def merge_spm(unmerged: SPMPrediction, config: SPMConfig = SPMConfig(), merge_only_seam: bool = False) -> SPMPrediction:
    merger = SpatialPolygonMerger(config)
    merger.index(unmerged)
    if merge_only_seam:
        return merger.merge(unmerged.seam_prediction_idxs)
    else:
        return merger.merge()


def merge_smm(unmerged: SPMPrediction, **params) -> SPMPrediction:
    from spatial_mask_merging.smm.smm import SpatialMaskMerger
    from utils.adapters import PredictionAdapter

    height, width = _frame_shape(unmerged)
    smm_prediction = PredictionAdapter(unmerged).smm
    merged_objects = SpatialMaskMerger(**params).merge(smm_prediction, image_size_hw=(height, width))

    result = SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape)
    for obj in merged_objects:
        _add_mask_annotation(result, obj["mask"], obj["label"], 0, obj["score"])
    return result


def merge_sahi(unmerged: SPMPrediction, match_threshold: float = 0.5,
               match_metric: str = "IOU", merger: str = "NMM") -> SPMPrediction:
    from sahi.postprocess.combine import GreedyNMMPostprocess, NMMPostprocess, NMSPostprocess, LSNMSPostprocess
    from sahi.postprocess.backends import set_postprocess_backend
    from utils.adapters import PredictionAdapter, SAHIPrediction

    set_postprocess_backend("numpy") # Set postprocess backend to numpy to avoid GPU memory usage during merging

    postprocessors = {"NMM": NMMPostprocess, "GREEDYNMM": GreedyNMMPostprocess, "NMS": NMSPostprocess, "LSNMS": LSNMSPostprocess}
    postprocess = postprocessors[merger.upper()](
        match_threshold=match_threshold, match_metric=match_metric, class_agnostic=False)

    sahi_prediction = PredictionAdapter(unmerged).sahi
    merged = postprocess(sahi_prediction.predictions)

    adapter = PredictionAdapter(SAHIPrediction(predictions=merged))
    result = adapter.spm
    result.image_path = unmerged.image_path
    result.image_shape = unmerged.image_shape
    return result


def merge_supervision(unmerged: SPMPrediction, iou_threshold: float = 0.5,
                      merge: bool = True) -> SPMPrediction:
    import supervision as sv

    height, width = _frame_shape(unmerged)
    if unmerged.count == 0:
        return SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape)

    masks = np.stack([_rasterize(p, height, width).astype(bool) for p in unmerged.polygons])
    detections = sv.Detections(
        xyxy=np.asarray(unmerged.bboxes, dtype=np.float32),
        mask=masks,
        confidence=np.asarray(unmerged.confidences, dtype=np.float32),
        class_id=np.asarray(unmerged.class_ids, dtype=int),
    )
    detections = detections.with_nmm(iou_threshold) if merge else detections.with_nms(iou_threshold)

    result = SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape)
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        name = unmerged.names[unmerged.class_ids.index(class_id)] if class_id in unmerged.class_ids else str(class_id)
        _add_mask_annotation(result, detections.mask[i].astype(np.uint8), name, class_id, detections.confidence[i])
    return result


MERGING_METHODS: dict[str, MergingMethod] = {
    "spm_seam_only": MergingMethod("spm_seam_only", merge_spm, merge_only_seam=True),
    "spm_global": MergingMethod("spm_global", merge_spm, merge_only_seam=False),
    "sahi_nms": MergingMethod("sahi_nms", merge_sahi, merger="NMS"),
    "sahi_nmm": MergingMethod("sahi_nmm", merge_sahi, merger="NMM"),
    "sahi_greedy_nmm": MergingMethod("sahi_greedy_nmm", merge_sahi, merger="GREEDYNMM"),
    "sahi_lsnms": MergingMethod("sahi_lsnms", merge_sahi, merger="LSNMS"),
    "smm": MergingMethod("smm", merge_smm)
}
