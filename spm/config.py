from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import rasterio
from shapely import Polygon

from spm.utils import xy_mask_to_polygon


@dataclass
class SPMPrediction:
    """Container for predictions from the model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""
    names: list[str] = field(default_factory=list)
    segmentations: list[list[tuple[float, float]]
                        ] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]
                 ] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    image_path: Path = None

    def add_annotation(self, name: str = None, class_id: int = None, confidence: float = None, bbox: tuple[float, float, float, float] = None, segmentation: list[tuple[float, float]] = None, polygon: Polygon = None):
        if polygon is None:
            # Assuming segmentation is a list of lists of (x, y) tuples
            polygon = self._to_polygon(segmentation)
        # TODO: Having both segmentation and polygon is redundant, we should consider only keeping one of them to save memory.
        self.polygons.append(polygon)
        self.segmentations.append(segmentation)
        # TODO: Bounding box size is likely to change after polygon simplification, so we may want to recalculate the bbox based on the final polygon.
        self.bboxes.append(bbox)
        self.class_ids.append(class_id)
        self.confidences.append(confidence)
        self.names.append(name)

    def _to_polygon(self, mask: list[tuple[float, float]]) -> Polygon:
        return xy_mask_to_polygon(mask)

    def _pixel_to_geo(self, transform, x, y):
        """
        Convert pixel coordinates to geographic coordinates.

        Parameters
        ----------
        transform : Affine
            Affine transformation from rasterio.
        x : int
            Pixel x-coordinate.
        y : int
            Pixel y-coordinate.

        Returns
        -------
        geo_x, geo_y : float
            Geographic coordinates.
        """
        geo_x, geo_y = transform * (x, y)
        return geo_x, geo_y

    def _to_geojson(self) -> dict:
        """Convert the SPMPrediction to a GeoJSON-like dictionary."""

        # TODO: This can function can be avoided by ensuring JSON serializability at the point of data creation, rather than having to convert it at the end.
        def make_json_serializable(value):
            import torch
            import numpy as np

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    return value.item()
                return value.tolist()

            if isinstance(value, np.ndarray):
                return value.tolist()

            if isinstance(value, np.generic):
                return value.item()

            return value

        with rasterio.open(self.image_path) as src:
            transform = src.transform
            crs = src.crs
        _crs = crs.to_string() if crs else "EPSG:4326"

        features = []

        for i in range(len(self.polygons)):
            feature = {
                "type": "Feature",
                "properties": {
                    "name": make_json_serializable(self.names[i]),
                    "class_id": make_json_serializable(self.class_ids[i]),
                    "confidence": make_json_serializable(self.confidences[i]),
                    # "bbox": make_json_serializable(self.bboxes[i]),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            self._pixel_to_geo(transform, float(x), float(y))
                            for x, y in self.polygons[i].exterior.coords
                        ]
                    ],
                },
            }

            features.append(feature)

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": _crs
                },
            },
            "features": features,
        }

    def save_as_geojson(self) -> None:
        """Save the SPMPrediction as a GeoJSON file."""
        geojson_dict = self._to_geojson()
        import json
        output_path = self.image_path.with_suffix(
            ".geojson") if self.image_path else Path("prediction.geojson")
        with open(output_path, "w") as f:
            json.dump(geojson_dict, f, indent=2)

    @property
    def count(self):
        """Returns the number of annotations."""
        return len(self.class_ids)


@dataclass
class SPMConfig:
    """Configuration for the Spatial Polygon Merging (SPM) algorithm."""
    tau_dist: float = 4.0  # Distance around geometry within which to query the STRtree for neighbors
    # Distance threshold for merging polygons (used in should_merge)
    rho_dist: float = 5.0
