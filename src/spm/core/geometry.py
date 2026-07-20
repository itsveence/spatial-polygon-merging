from typing import Optional, Union

from shapely import MultiPolygon, Polygon

from spm.utils.logger import logger


def effective_overlap(
    tile_x: int,
    tile_y: int,
    tile_size: int,
    overlap: int,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> int:
    """Largest actual tile-to-neighbor overlap for the tile at ``(tile_x, tile_y)``.

    Tile streaming clamps the last tile in each axis to fit the image, which makes
    that tile's seam with its predecessor wider than the canonical ``overlap``.
    Pass the value returned here as the ``overlap`` argument to
    ``is_index_candidate`` so border candidates inside those widened seams are not
    missed. Returns ``overlap`` for tiles with only canonical neighbors, and 0 for
    a tile with no neighbors at all.
    """
    step = tile_size - overlap
    side_overlaps = []

    def collect(tile_pos: int, axis_size: int | None) -> None:
        # Previous tile sits at a canonical step position (only the last tile
        # in an axis is ever clamped, and it is never a left/top neighbor).
        if tile_pos > 0:
            prev_left = ((tile_pos - 1) // step) * step
            side_overlaps.append((prev_left + tile_size) - tile_pos)
        # Next tile may be clamped to (axis_size - tile_size) when its canonical
        # position would overshoot the image.
        if axis_size is None or (tile_pos + tile_size) < axis_size:
            next_canonical = tile_pos + step
            if axis_size is not None and next_canonical + tile_size > axis_size:
                next_left = axis_size - tile_size
            else:
                next_left = next_canonical
            side_overlaps.append((tile_pos + tile_size) - next_left)

    collect(tile_x, img_width)
    collect(tile_y, img_height)

    return max(side_overlaps) if side_overlaps else 0


def is_overlap_candidate(
    bbox: tuple[float, float, float, float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    overlap: int,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> bool:
    """Determines if a bounding box is an overlap candidate.

    ``bbox`` must be in tile-local coordinates (as produced by a model run on the
    tile). A bbox is an overlap candidate if it lies within ``overlap`` of a tile
    edge that has a neighboring tile on that side. Tiles at the image boundary
    have no neighbor on that side and are excluded from that edge.
    """
    x_min, y_min, x_max, y_max = bbox

    near_left = tile_x != 0 and x_min <= overlap
    near_top = tile_y != 0 and y_min <= overlap
    near_right = (tile_x + tile_size) != img_width and x_max >= (tile_size - overlap)
    near_bottom = (tile_y + tile_size) != img_height and y_max >= (tile_size - overlap)

    logger.debug(
        f"Checking bbox {bbox} size={tile_size} overlap={overlap}: "
        f"near_left={near_left}, near_top={near_top}, near_right={near_right}, near_bottom={near_bottom}"
    )

    return near_left or near_top or near_right or near_bottom


def xy_mask_to_polygon(
    mask: list[list[tuple[float, float]]],
) -> Union[Polygon, MultiPolygon]:
    """Converts a list of (x, y) coordinates representing a mask into a Shapely Polygon."""
    if len(mask) > 1:
        polygon = MultiPolygon([Polygon(m) for m in mask])
    else:
        polygon = Polygon(mask[0])
    return polygon
