import math

import numpy as np
from shapely import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from utils.logger import logger

def effective_overlap(
        tile_x: int,
        tile_y: int,
        tile_size: int,
        overlap: int,
        img_width: int = None,
        img_height: int = None,
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


def is_index_candidate(
        bbox: tuple[float, float, float, float],
        tile_size: int,
        overlap: int
        ) -> bool:
    """Determines if a bounding box is a border candidate for indexing.

    ``bbox`` must be in tile-local coordinates (as produced by a model run on the
    tile). A bbox is a border candidate if it lies within ``overlap`` of a tile
    edge that has a neighboring tile on that side. Tiles at the image boundary
    have no neighbor on that side and are excluded from that edge.
    """
    x_min, y_min, x_max, y_max = bbox

    near_left   = x_min <= overlap
    near_top    = y_min <= overlap
    near_right  = x_max >= (tile_size - overlap)
    near_bottom = y_max >= (tile_size - overlap)

    logger.debug(
        f"Checking bbox {bbox} size={tile_size} overlap={overlap}: "
        f"near_left={near_left}, near_top={near_top}, near_right={near_right}, near_bottom={near_bottom}"
    )

    return near_left or near_top or near_right or near_bottom

def _ensure_valid(poly: Polygon) -> Polygon:
        """
        Return a valid Shapely geometry, repairing self-intersections if needed.
        make_valid may return a GeometryCollection for badly degenerate inputs,
        so we extract only Polygon/MultiPolygon parts.
        """
        if poly.is_valid:
            return poly
        fixed = make_valid(poly)
        # make_valid can produce GeometryCollection — keep only area-bearing parts
        if fixed.geom_type == "GeometryCollection":
            polys = []
            for g in fixed.geoms:
                if isinstance(g, MultiPolygon):
                    polys.extend(g.geoms)
                else:
                    polys.append(g)
            if not polys:
                return Polygon()  # empty sentinel — will be ignored downstream
            fixed = unary_union(polys)
            fixed = fixed.buffer(5.0).buffer(-5.0)
        return fixed

def xy_mask_to_polygon(mask: list[tuple[float, float]]) -> Polygon:
    """Converts a list of (x, y) coordinates representing a mask into a Shapely Polygon."""
    polygon = Polygon(mask)
    return sanitize_polygon(polygon)

def sanitize_polygon(poly: Polygon, simplify_tolerance: float = 5, angle_threshold_deg: float = 5.0) -> Polygon | None:
    """
    Fix common GEOS topology issues:
    - Duplicate/zero-length vertices
    - Self-intersections
    - Invalid rings
    """
    if poly is None or poly.is_empty:
        return None

    # Remove duplicate consecutive coords (zero-length edges)
    coords = list(poly.exterior.coords)
    deduped = [coords[0]]
    for c in coords[1:]:
        if c != deduped[-1]:  # skip exact duplicates
            deduped.append(c)

    # Need at least 4 coords to form a valid ring
    if len(deduped) < 4:
        return None

    poly = Polygon(deduped)

    # Simplify to remove near-duplicate vertices causing
    # non-noded intersections (tolerance in pixels)
    poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    if poly.is_empty:
        return None

    # Fix any remaining invalidity (self-touches, self-crosses)
    if not poly.is_valid:
        poly = make_valid(poly)

    # make_valid can return GeometryCollection — extract largest polygon
    if poly.geom_type == 'GeometryCollection':
        polys = [g for g in poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        poly = max(polys, key=lambda g: g.area)

    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda g: g.area)

    if poly.is_empty or not isinstance(poly, Polygon):
        return None
    
    # Turn-angle pruning
    coords = list(poly.exterior.coords[:-1])  # drop the closing duplicate
    n = len(coords)
    if n < 3:
        return poly

    thresh = math.radians(angle_threshold_deg)
    keep = []
    for i in range(n):
        a = np.array(coords[(i - 1) % n])
        b = np.array(coords[i])
        c = np.array(coords[(i + 1) % n])
        ab = b - a
        bc = c - b
        lab, lbc = np.linalg.norm(ab), np.linalg.norm(bc)
        if lab < 1e-9 or lbc < 1e-9:
            continue  # degenerate zero-length edge — skip
        cos_t = np.clip(np.dot(ab, bc) / (lab * lbc), -1.0, 1.0)
        # arccos gives the angle between consecutive edge vectors:
        #   0°  → perfectly straight (collinear)   → prune
        #   90° → right-angle corner               → keep
        if np.arccos(cos_t) >= thresh:
            keep.append(coords[i])

    if len(keep) < 3:
        return poly

    return Polygon(keep)