from __future__ import annotations

import multiprocessing as mp
import os
import queue as mp_queues
import threading
import time
from typing import Callable
import resource, functools

import cv2
import numpy as np
import psutil

from spm.config.config import SPMConfig
from spm.core.prediction import SPMPrediction
from spm.merging.spm import SpatialPolygonMerger
from spm.preprocessing.image_processing import binary_mask_to_contours
from spm.utils.logger import logger

MergeFn = Callable[..., SPMPrediction]

# RSS poll interval for the in-child memory monitor. Small enough to catch
# short native allocation spikes that a coarse sampler would miss.
_RSS_POLL_INTERVAL = 0.001

def memory_limit(max_gb):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            max_bytes = int(max_gb * 1024**3)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
            try:
                return func(*args, **kwargs)
            finally:
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))  # restore
        return wrapper
    return decorator

def _sample_peak_rss(proc: psutil.Process, stop: threading.Event, out: list) -> None:
    """Track the high-water RSS of ``proc`` until ``stop`` is set."""
    peak = proc.memory_info().rss
    while not stop.is_set():
        try:
            peak = max(peak, proc.memory_info().rss)
        except psutil.Error:
            break
        time.sleep(_RSS_POLL_INTERVAL)
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
        merged, merge_time = fn(unmerged, *args, **kwargs)
    except Exception as exc:
        # Ship the failure to the parent instead of dying silently, which
        # would leave the parent blocked on queue.get() forever.
        queue.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
        return
    finally:
        stop.set()
        monitor.join()

    peak_bytes = max(peak_holder[0] - baseline, 0)
    queue.put((merged, merge_time, peak_bytes / (1024 * 1024)))


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
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        worker = ctx.Process(
            target=_merge_worker,
            args=(self._fn, unmerged, self._args, self._kwargs, queue),
        )
        worker.start()
        # Poll rather than block: if the child is killed outright (e.g. the
        # kernel OOM killer), nothing is ever put on the queue.
        while True:
            try:
                payload = queue.get(timeout=1.0)
                break
            except mp_queues.Empty:
                if not worker.is_alive():
                    raise RuntimeError(
                        f"merge worker for '{self.name}' died with exit code "
                        f"{worker.exitcode} without returning a result")
        worker.join()
        if isinstance(payload, Exception):
            raise payload
        merged, merge_time, peak_memory_usage = payload

        merged.image_path = unmerged.image_path
        merged.image_shape = unmerged.image_shape
        return merged, merge_time, peak_memory_usage


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


@memory_limit(40)
def merge_spm(unmerged: SPMPrediction, config: SPMConfig = SPMConfig(), merge_only_seam: bool = False) -> tuple[SPMPrediction, float]:
    logger.info(f"Running SPM with config: {config}, merge_only_seam={merge_only_seam}")
    merger = SpatialPolygonMerger(config)
    merger.index(unmerged)
    start_time = time.perf_counter()
    if merge_only_seam:
        merged =  merger.merge(unmerged.seam_prediction_idxs)
    else:
        merged = merger.merge()
    merge_time = time.perf_counter() - start_time
    return merged, merge_time


@memory_limit(40)
def merge_smm(unmerged: SPMPrediction, **params) -> tuple[SPMPrediction, float]:
    from spatial_mask_merging.smm.smm import SpatialMaskMerger
    from utils.adapters import PredictionAdapter

    logger.info(f"Running SMM with parameters: {params}")
    height, width = _frame_shape(unmerged)
    smm_prediction = PredictionAdapter(unmerged).smm
    smm = SpatialMaskMerger(
        tau_d=params.get("tau_d", 5.0),
        tau_i=params.get("tau_i", 0.5),
        rho=params.get("rho", 10.0),
        beta1=params.get("beta1", 0.3),
        beta2=params.get("beta2", 0.5),
        beta3=params.get("beta3", 0.2),
        gamma=params.get("gamma", 0.5),
        lambda_=params.get("lambda_", 0.5)
        )
    start_time = time.perf_counter()
    merged_objects = smm.merge(smm_prediction, image_size_hw=(height, width))
    merge_time = time.perf_counter() - start_time

    result = SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape)
    for obj in merged_objects:
        _add_mask_annotation(result, obj["mask"], obj["label"], 0, obj["score"])
    return result, merge_time


@memory_limit(40)
def merge_sahi(unmerged: SPMPrediction, match_threshold: float = 0.3,
               match_metric: str = "IOU", merger: str = "NMM") -> tuple[SPMPrediction, float]:
    from sahi.postprocess.combine import GreedyNMMPostprocess, NMMPostprocess, NMSPostprocess, LSNMSPostprocess
    from sahi.postprocess.backends import set_postprocess_backend
    from utils.adapters import PredictionAdapter, SAHIPrediction

    logger.info(f"Running SAHI with parameters: match_threshold={match_threshold}, match_metric={match_metric}, merger={merger}")
    set_postprocess_backend("numpy") # Set postprocess backend to numpy to avoid GPU memory usage during merging

    postprocessors = {"NMM": NMMPostprocess, "GREEDYNMM": GreedyNMMPostprocess, "NMS": NMSPostprocess, "LSNMS": LSNMSPostprocess}
    postprocess = postprocessors[merger.upper()](
        match_threshold=match_threshold, match_metric=match_metric, class_agnostic=False)

    sahi_prediction = PredictionAdapter(unmerged).sahi

    start_time = time.perf_counter()
    merged = postprocess(sahi_prediction.predictions)
    merge_time = time.perf_counter() - start_time

    adapter = PredictionAdapter(SAHIPrediction(predictions=merged))
    result = adapter.spm
    result.image_path = unmerged.image_path
    result.image_shape = unmerged.image_shape
    return result, merge_time


@memory_limit(40)
def merge_supervision(unmerged: SPMPrediction, iou_threshold: float = 0.5,
                      merge: bool = True) -> tuple[SPMPrediction, float]:
    import supervision as sv

    height, width = _frame_shape(unmerged)
    if unmerged.count == 0:
        return SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape), 0.0

    masks = np.stack([_rasterize(p, height, width).astype(bool) for p in unmerged.polygons])
    detections = sv.Detections(
        xyxy=np.asarray(unmerged.bboxes, dtype=np.float32),
        mask=masks,
        confidence=np.asarray(unmerged.confidences, dtype=np.float32),
        class_id=np.asarray(unmerged.class_ids, dtype=int),
    )

    start_time = time.perf_counter()
    detections = detections.with_nmm(iou_threshold) if merge else detections.with_nms(iou_threshold)
    merge_time = time.perf_counter() - start_time

    result = SPMPrediction(image_path=unmerged.image_path, image_shape=unmerged.image_shape)
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        name = unmerged.names[unmerged.class_ids.index(class_id)] if class_id in unmerged.class_ids else str(class_id)
        _add_mask_annotation(result, detections.mask[i].astype(np.uint8), name, class_id, detections.confidence[i])
    return result, merge_time

# smm_params = {
#         "tau_d":5.0,      # Distance threshold (pixels)
#         "tau_i":0.5,       # IoU threshold
#         "rho":10.0,        # R-tree search radius (pixels)
#         "beta1":0.3,       # Distance weight
#         "beta2":0.5,       # IoU weight
#         "beta3":0.2,       # Confidence weight
#         "gamma":0.5,       # Anti-chaining threshold
#         "lambda_":0.5      # Clustering penalty
#     }

# Optimized SMM parameters from hyperparameter tuning
smm_params = {
  "tau_d": 21.855988093291327,
  "tau_i": 0.40396342434325805,
  "rho": 44.6431390422366,
  "beta1": 0.3406807500462869,
  "beta2": 0.5791467587819955,
  "beta3": 0.26279874111556634,
  "gamma": 0.43312966514646717,
  "lambda_": 1.0870328415148245
}

MERGING_METHODS: dict[str, MergingMethod] = {
    "spm_seam_only": MergingMethod("spm_seam_only", merge_spm, merge_only_seam=True),
    "spm_global": MergingMethod("spm_global", merge_spm, merge_only_seam=False),
    "sahi_nms": MergingMethod("sahi_nms", merge_sahi, merger="NMS"),
    "sahi_nmm": MergingMethod("sahi_nmm", merge_sahi, merger="NMM"),
    "sahi_greedy_nmm": MergingMethod("sahi_greedy_nmm", merge_sahi, merger="GREEDYNMM"),
    "sahi_lsnms": MergingMethod("sahi_lsnms", merge_sahi, merger="LSNMS"),
    "smm": MergingMethod("smm", merge_smm, **smm_params),
}
