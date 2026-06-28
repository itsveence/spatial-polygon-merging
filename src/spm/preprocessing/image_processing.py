from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
import rasterio
from typing import Generator, Union

from spm.utils.logger import logger
from spm.utils.profiling import time_it


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


def binary_mask_to_contours(mask: np.ndarray, min_area: int = 20, approx_contour: bool = False, contour_approx_factor: float = 0.01, normalize: bool = True, sort_by_area: bool = False) -> list[list[Union[float, int]]]:
    """Converts binary masks to contours, with options for filtering by area, normalizing coordinates, and sorting by area.

    Args:
        mask (np.ndarray): Binary mask where the object pixels are 255/True and the background is 0/False.
        min_area (int, optional): Minimum area of contours to include. Defaults to 20.
        approx_contour (bool, optional): Whether to approximate contours. Defaults to False.
        contour_approx_factor (float, optional): Factor for approximating contours. Defaults to 0.01.
        normalize (bool, optional): Whether to normalize coordinates. Defaults to True.
        sort_by_area (bool, optional): Whether to sort contours by area. Defaults to False.

    Returns:
        list[list[Union[float, int]]]: List of contours, where each contour is a list of (x, y) coordinates. 
        Coordinates are normalized to [0, 1] if `normalize` is True, otherwise they are in pixel values. 
        Contours with area less than `min_area` are excluded. If `sort_by_area` is True, contours are sorted in descending order of area.
    """

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} contours in mask")

    h, w = mask.shape

    final_contours = []
    list_of_areas = []  # For sorting contours by area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        if approx_contour:
            epsilon = contour_approx_factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2)

        if normalize:
            normalized = (points/[w, h]).ravel().tolist()
            logger.debug(
                f"Contour with area {area} has {len(points)} points, normalized: {normalized}")
            final_contours.append(normalized)
        else:
            contour = points.ravel().tolist()
            logger.debug(
                f"Contour with area {area} has {len(points)} points, raw: {contour}")
            final_contours.append(contour)

        list_of_areas.append(area)
        
    if sort_by_area:
        sorted_indexes = sorted(range(len(list_of_areas)),
                                key=list_of_areas.__getitem__, reverse=True)
        return [final_contours[i] for i in sorted_indexes]
    else:
        return final_contours


def mask_file_to_polygons(mask_path: Path, min_area: int = 20, normalize: bool = True) -> list[list[Union[float, int]]]:
    """Converts binary mask to list of polygons."""
    mask = tiff.imread(mask_path)
    logger.debug(
        f"Loaded mask from {mask_path} with shape {mask.shape} and dtype {mask.dtype}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = ~mask  # Invert mask: 0/False becomes 255/True, and vice versa

    binary = mask.astype(np.uint8)
    polygons = binary_mask_to_contours(
        binary, min_area=min_area, normalize=normalize)

    return polygons


@time_it
def mask_to_txt(mask_path: Path, txt_path: Path, min_area: int = 20) -> None:
    """Converts mask images to YOLO text format."""
    try:
        polygons = mask_file_to_polygons(
            mask_path, min_area=min_area, normalize=True)
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
    """Streams tiles from a raster source in batches, with options for tile size, batch size, and overlap.

    Args:
        src (rasterio.DatasetReader): _description_
        tile_size (int, optional): _description_. Defaults to 640.
        batch_size (int, optional): _description_. Defaults to 4.
        overlap (int, optional): _description_. Defaults to 0.

    Yields:
        Generator[tuple[list[np.ndarray], list[tuple[int, int]]], None, None]: A generator that yields tuples containing a list of image tiles (as numpy arrays) and a corresponding list of their coordinates (top, left, bottom, right).
    """
    width, height = src.width, src.height

    # If the tile size is larger than the image height or width, then return the whole image as one tile, with the 4 coordinates
    if tile_size >= min(height, width):
        logger.warning(
            f"Tile size {tile_size} is larger than image dimensions {height}x{width}. Returning the whole image as one tile.")
        img = src.read().transpose(1, 2, 0)  # Read the whole image
        img_bgr = img[..., ::-1]  # flip to BGR for OpenCV/Ultralytics
        # Yield the whole image as one tile with its coordinates
        yield [img_bgr], [(0, 0, height, width)]
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
