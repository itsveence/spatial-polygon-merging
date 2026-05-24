from shapely import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from utils.logger import logger

def is_index_candidate(
        bbox: tuple[float, float, float, float],
        tile_x: int,
        tile_y: int,
        tile_size: int,
        overlap: int,
        img_width: int = None,
        img_height: int = None,
        ) -> bool:
    """Determines if a bounding box is a candidate for indexing based on its position relative to the tile boundaries."""
    x_min, y_min, x_max, y_max = bbox

    # Adjust bbox coordinates to the tile's coordinate system
    x_min = x_min + tile_x
    x_max = x_max + tile_x
    y_min = y_min + tile_y
    y_max = y_max + tile_y

    tile_left = tile_x
    tile_top = tile_y
    tile_right = tile_x + tile_size
    tile_bottom = tile_y + tile_size

    near_right  = (img_width is None or tile_right  < img_width)    and (x_max >= tile_right  - overlap)
    near_left   = (tile_x > 0)                                      and (x_min <= tile_left   + overlap)
    near_bottom = (img_height is None or tile_bottom < img_height)  and (y_max >= tile_bottom - overlap)
    near_top    = (tile_y > 0)                                      and (y_min <= tile_top    + overlap)

    logger.debug(
        f"Checking bbox {bbox} against tile ({tile_left}, {tile_top}, {tile_right}, {tile_bottom}) with overlap {overlap}: "
        f"near_right={near_right}, near_left={near_left}, near_bottom={near_bottom}, near_top={near_top}"
    )

    return near_right or near_left or near_top or near_bottom

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
            polys = [g for g in fixed.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return Polygon()  # empty sentinel — will be ignored downstream
            fixed = unary_union(polys)
        return fixed

def xy_mask_to_polygon(mask: list[tuple[float, float]]) -> Polygon:
    """Converts a list of (x, y) coordinates representing a mask into a Shapely Polygon."""
    polygon = Polygon(mask)
    if not polygon.is_valid:
        logger.warning("Polygon is invalid, attempting to fix with make_valid.")
        polygon = _ensure_valid(polygon)
    return polygon