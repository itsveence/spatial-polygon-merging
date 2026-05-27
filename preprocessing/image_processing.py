from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
import rasterio
from typing import Generator, Union

import torch
from spm.config import SPMPrediction
from spm.spm import SpatialPolygonMerger
from spm.utils import effective_overlap, is_border_candidate
from utils.logger import logger
from utils.helpers import time_it

def tiff_to_png(tiff_path: Path, png_path: Path) -> None:
    """Converts TIFF images to PNG format."""
    try:
        img = tiff.imread(tiff_path)
        
        # Convert GRAY to RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Drop alpha channel if present

        cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.error(f"Failed to convert {tiff_path} to PNG: {e}")

def binary_mask_to_polygon(mask_path: Path, min_area: int = 20, normalize: bool = True) -> list[list[Union[float, int]]]:
    """Converts binary mask to list of polygons."""
    mask = tiff.imread(mask_path)
    logger.debug(f"Loaded mask from {mask_path} with shape {mask.shape} and dtype {mask.dtype}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = ~mask # Invert mask: 0/False becomes 255/True, and vice versa

    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} contours in {mask_path}: {contours}")

    h, w = binary.shape
    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        points = approx.reshape(-1, 2)

        if normalize:
            # normalize x,y coordinates
            normalized = (points/[w, h]).ravel().tolist()
            logger.debug(f"Contour with area {area} has {len(points)} points, normalized: {normalized}")
            polygons.append(normalized)
        else:
            polygon = points.ravel().tolist()
            logger.debug(f"Contour with area {area} has {len(points)} points, raw: {polygon}")
            polygons.append(polygon)

    return polygons

@time_it
def mask_to_txt(mask_path: Path, txt_path: Path, min_area: int = 20) -> None:
    """Converts mask images to YOLO text format."""
    try:
        polygons = binary_mask_to_polygon(mask_path, min_area=min_area, normalize=True)
        lines = []
        for poly in polygons:
            line = f"{0} " + " ".join(f"{p:.6f}" for p in poly)
            lines.append(line)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to convert {mask_path} to TXT: {e}")

def read_tile(src, top, left, bottom, right):
    """Reads a tile from the raster source."""
    height = bottom - top
    width = right - left
    window = rasterio.windows.Window.from_slices((top, bottom), (left, right))

    tile = src.read(
        window=window,
        boundless=True,
        fill_value=0,
        masked=True,
        out_shape=(src.count, height, width),
    )

    img_np = np.asarray(tile.filled(0)).astype(np.uint8)
    img_rgb = img_np[:3].transpose(1, 2, 0)
    img_bgr = img_rgb[..., ::-1]  # flip to BGR for OpenCV/Ultralytics
    return img_bgr


def stream_tiles_by_batch(src: rasterio.DatasetReader, tile_size: int = 640, batch_size: int = 4, overlap: int = 0) -> Generator[tuple[list[np.ndarray], list[tuple[int, int]]], None, None]:
    """Streams image tiles in batches."""
    width, height = src.width, src.height

    #If the tile size is larger than the image height or width, then return the whole image as one tile, with the 4 coordinates
    if tile_size >= min(height, width):
        logger.warning(f"Tile size {tile_size} is larger than image dimensions {height}x{width}. Returning the whole image as one tile.")
        img = src.read().transpose(1, 2, 0)  # Read the whole image
        img_bgr = img[..., ::-1]  # flip to BGR for OpenCV/Ultralytics
        yield [img_bgr], [(0, 0, height, width)]  # Yield the whole image as one tile with its coordinates
        return
    
    tiles = []
    coordinates = []
    step = tile_size - overlap
    for top in range(0, height - tile_size + step, step):
        bottom = top + tile_size
        # Adjust for boundaries: if the tile goes beyond the image height, shift it upwards while keeping the tile size constant
        bottom = min(bottom, height)
        top = max(0, bottom - tile_size)

        for left in range(0, width - tile_size + step, step):
            right = left + tile_size
            # Adjust for boundaries: if the tile goes beyond the image width, shift it left while keeping the tile size constant
            right = min(right, width)
            left = max(0, right - tile_size)

            # Read the tile data and append to the batch along with its coordinates
            tile = read_tile(src, top, left, bottom, right)
            tiles.append(tile)
            coordinates.append((top, left, bottom, right))

            # Once we have enough tiles for a batch, yield them and reset the lists
            if len(tiles) == batch_size:
                yield tiles, coordinates
                tiles = []
                coordinates = []
    if tiles:
        yield tiles, coordinates  # Yield remaining tiles if any
