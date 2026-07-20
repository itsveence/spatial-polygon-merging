from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import rasterio
from shapely import Polygon
import geopandas as gpd

from spm.core.geometry import xy_mask_to_polygon
from spm.io.raster import get_crs_and_transform, pixel_to_geo
from spm.utils.logger import logger
from typing import Optional


@dataclass
class SPMPrediction:
    """Container for predictions from the model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""

    names: list[str] = field(default_factory=list)
    segmentations: list[list[list[tuple[float, float]]]] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    image_path: Optional[Path] = None
    image_shape: Optional[tuple[int, int]] = None  # (height, width)

    seam_prediction_idxs: list[int] = field(default_factory=list)

    def add_annotation(
        self,
        name: Optional[str] = None,
        class_id: Optional[int] = None,
        confidence: Optional[float] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        segmentation: Optional[list[list[tuple[float, float]]]] = None,
        polygon: Optional[Polygon] = None,
    ):
        if polygon is None:
            # Assuming segmentation is a list of lists of (x, y) tuples
            if segmentation is None:
                raise ValueError("Either polygon or segmentation must be provided.")
            polygon = self._to_polygon(segmentation)
        if segmentation is None:
            segmentation = []
        if class_id is None:
            raise ValueError("class_id must be provided.")
        if confidence is None:
            raise ValueError("confidence must be provided.")
        if name is None:
            raise ValueError("name must be provided.")
        # TODO: Having both segmentation and polygon is redundant, we should consider only keeping one of them to save memory.
        self.polygons.append(polygon)
        self.segmentations.append(segmentation)
        # TODO: Bounding box size is likely to change after polygon simplification, so we may want to recalculate the bbox based on the final polygon.
        self.bboxes.append(polygon.bounds)
        self.class_ids.append(class_id)
        self.confidences.append(confidence)
        self.names.append(name)

    def _to_polygon(self, mask: list[list[tuple[float, float]]]) -> Polygon:
        return xy_mask_to_polygon(mask)

    def _get_crs_and_transform(self):
        if self.image_path is None:
            raise ValueError("image_path is not set for SPMPrediction.")
        with rasterio.open(self.image_path) as src:
            return src.crs, src.transform

    def _to_gdf(self) -> gpd.GeoDataFrame:
        """Convert the SPMPrediction to a GeoPandas GeoDataFrame."""
        from shapely.ops import transform

        crs, affine_transform = get_crs_and_transform(self.image_path)

        def pixel_to_geo(x, y, z=None):
            x_geo, y_geo = affine_transform * (x, y)
            return x_geo, y_geo

        geometries = [transform(pixel_to_geo, geom) for geom in self.polygons]

        properties = {
            "name": self.names,
            "class_id": self.class_ids,
            "confidence": self.confidences,
            "bbox": self.bboxes,
        }

        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)

        return gdf

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

        crs, transform = get_crs_and_transform(self.image_path)
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
                            pixel_to_geo(transform, float(x), float(y))
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
                "properties": {"name": _crs},
            },
            "features": features,
        }

    def to_json(self) -> dict:
        """Convert the SPMPrediction to a dictionary."""
        if self.image_path is None:
            raise ValueError("image_path is not set for SPMPrediction.")
        with rasterio.open(self.image_path) as src:
            height, width = src.height, src.width

        annotations: list[dict] = []
        for i in range(len(self.polygons)):
            coords = [
                [int(round(px)), int(round(py))]
                for px, py in self.polygons[i].exterior.coords
            ]
            if len(coords) < 3:
                continue
            minx, miny, maxx, maxy = self.polygons[i].bounds
            bbox = [
                int(round(minx)),
                int(round(miny)),
                int(round(maxx)),
                int(round(maxy)),
            ]
            annotations.append(
                {
                    "type": "object",
                    "class_id": self.class_ids[i],
                    "bbox": bbox,
                    "area": float(self.polygons[i].area),
                    "segmentation": [coords],
                    "confidence": self.confidences[i],
                }
            )

        return {
            "image_name": Path(self.image_path).name,
            "image_size": (height, width),
            "annotations": annotations,
        }

    def save_to_file(
        self,
        format: str = "gpkg",
        name: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save the SPMPrediction as a GeoJSON or GPKG file."""
        format = format.lower()

        if format not in ["geojson", "gpkg"]:
            raise ValueError(f"Unsupported format: {format}")

        dir = (
            Path(f"predictions/{format}")
            if not output_dir
            else Path(output_dir) / format
        )
        dir.mkdir(parents=True, exist_ok=True)

        file_name = (
            Path(f"{name}.{format}")
            if name
            else (
                Path(f"{self.image_path.stem}.{format}")
                if self.image_path
                else Path(f"prediction.{format}")
            )
        )

        output_path = dir / file_name

        if format == "geojson":
            geojson_dict = self._to_geojson()
            import json

            with open(output_path, "w") as f:
                json.dump(geojson_dict, f, indent=2)

        elif format == "gpkg":
            gdf = self._to_gdf()
            gdf.to_file(output_path, driver="GPKG")

        logger.info(f"Saved prediction to {output_path}")

    @property
    def count(self):
        """Returns the number of annotations."""
        return len(self.class_ids)

    @property
    def seam_predictions(self) -> SPMPrediction:
        """Return a new SPMPrediction containing only the seam predictions."""
        if not self.seam_prediction_idxs:
            return SPMPrediction()

        seam_pred = SPMPrediction(
            names=[self.names[i] for i in self.seam_prediction_idxs],
            segmentations=[self.segmentations[i] for i in self.seam_prediction_idxs],
            polygons=[self.polygons[i] for i in self.seam_prediction_idxs],
            bboxes=[self.bboxes[i] for i in self.seam_prediction_idxs],
            class_ids=[self.class_ids[i] for i in self.seam_prediction_idxs],
            confidences=[self.confidences[i] for i in self.seam_prediction_idxs],
            image_path=self.image_path,
            image_shape=self.image_shape,
        )
        return seam_pred
