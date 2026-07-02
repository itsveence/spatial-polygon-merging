from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import rasterio
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box

from spm.utils.logger import logger


@dataclass(frozen=True)
class Crop:
    """A single generated test crop and its ground-truth labels."""
    crop_size: int
    index: int
    image_path: Path
    label_path: Path


def _windows(width: int, height: int, crop_size: int, stride: int) -> Iterable[Window]:
    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            yield Window(left, top, crop_size, crop_size)


def generate_test_set(
    image_path: str | Path,
    shapefile_path: str | Path,
    output_dir: str | Path,
    crop_sizes: Iterable[int],
    stride_ratio: float = 1.0,
    max_crops_per_size: int | None = None,
    min_labels: int = 1,
) -> list[Crop]:
    """Cut a large GeoTIFF and its label shapefile into test crops.

    For every size in ``crop_sizes`` the raster is tiled into square windows. Each
    window is written as ``crop_<size>_<i>/image.tif`` with the ground-truth polygons
    clipped to the window stored alongside as ``labels.gpkg`` (kept in the raster CRS).

    Args:
        image_path: Source GeoTIFF.
        shapefile_path: Polygon ground truth (.shp or any vector format geopandas reads).
        output_dir: Root directory for the generated test set.
        crop_sizes: Square crop sizes in pixels, e.g. ``[512, 1024, 2048]``.
        stride_ratio: Window step as a fraction of the crop size (1.0 = non-overlapping).
        max_crops_per_size: Cap on crops emitted per size; keeps the densest ones first.
        min_labels: Skip windows containing fewer than this many ground-truth polygons.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crops: list[Crop] = []

    with rasterio.open(image_path) as src:
        labels = gpd.read_file(shapefile_path).to_crs(src.crs)
        labels = labels[labels.geometry.notna() & ~labels.geometry.is_empty]
        invalid = ~labels.geometry.is_valid
        if invalid.any():
            logger.warning(f"Repairing {int(invalid.sum())} invalid label geometries")
            labels.loc[invalid, labels.geometry.name] = labels.geometry[invalid].make_valid()
            labels = labels[~labels.geometry.is_empty]
        sindex = labels.sindex

        for crop_size in crop_sizes:
            if crop_size > min(src.width, src.height):
                logger.warning(
                    f"Crop size {crop_size} exceeds image {src.width}x{src.height}; skipping.")
                continue

            stride = max(1, round(crop_size * stride_ratio))
            candidates: list[tuple[Window, gpd.GeoDataFrame]] = []

            for window in _windows(src.width, src.height, crop_size, stride):
                win_transform = window_transform(window, src.transform)
                left, top = win_transform * (0, 0)
                right, bottom = win_transform * (crop_size, crop_size)
                bounds = box(min(left, right), min(top, bottom),
                             max(left, right), max(top, bottom))

                hits = labels.iloc[list(sindex.intersection(bounds.bounds))]
                clipped = gpd.clip(hits, bounds)
                clipped = clipped[~clipped.geometry.is_empty]
                if len(clipped) < min_labels:
                    continue
                candidates.append((window, clipped))

            candidates.sort(key=lambda wc: len(wc[1]), reverse=True)
            if max_crops_per_size is not None:
                candidates = candidates[:max_crops_per_size]

            for index, (window, clipped) in enumerate(candidates):
                crop = _write_crop(src, window, clipped, output_dir, crop_size, index)
                crops.append(crop)
                logger.info(
                    f"Wrote crop {crop.image_path} with {len(clipped)} labels")

    logger.info(f"Generated {len(crops)} crops in {output_dir}")
    return crops


def _write_crop(
    src: rasterio.DatasetReader,
    window: Window,
    labels: gpd.GeoDataFrame,
    output_dir: Path,
    crop_size: int,
    index: int,
) -> Crop:
    crop_dir = output_dir / f"crop_{crop_size}_{index}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    image_path = crop_dir / "image.tif"
    label_path = crop_dir / "labels.gpkg"

    profile = src.profile.copy()
    profile.update(
        width=crop_size,
        height=crop_size,
        transform=window_transform(window, src.transform),
    )
    with rasterio.open(image_path, "w", **profile) as dst:
        dst.write(src.read(window=window))

    labels.to_file(label_path, driver="GPKG")
    return Crop(crop_size, index, image_path, label_path)
