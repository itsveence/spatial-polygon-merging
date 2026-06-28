from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import rasterio
from shapely import Polygon
import geopandas as gpd

from spm.utils import xy_mask_to_polygon
from sahi.prediction import ObjectPrediction
from spatial_mask_merging.smm.predictions import SMMPrediction

@dataclass
class SAHIPrediction:
    """Container for predictions from the SAHI model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""
    predictions: list[ObjectPrediction] = field(default_factory=list)

class PredictionAdapter:
    """Adapter class to handle conversion between different prediction objects [SPMPrediction, SMMPrediction, SAHIPrediction]"""

    def __init__(self, prediction: Union[SPMPrediction, SMMPrediction, SAHIPrediction]):

        assert isinstance(prediction, (SPMPrediction, SMMPrediction, SAHIPrediction)), "prediction must be an instance of SPMPrediction, SMMPrediction, or SAHIPrediction"

        self.spm_prediction = prediction if isinstance(prediction, SPMPrediction) else None
        self.smm_prediction = prediction if isinstance(prediction, SMMPrediction) else None
        self.sahi_prediction = prediction if isinstance(prediction, SAHIPrediction) else None

        # If the input is SMMPrediction or SAHIPrediction, convert it to SPMPrediction
        if isinstance(prediction, SMMPrediction):
            self.spm_prediction = self._smm_to_spm()
        elif isinstance(prediction, SAHIPrediction):
            self.spm_prediction = self._sahi_to_spm()

    @property
    def spm(self) -> SPMPrediction:
        if self.spm_prediction is not None:
            return self.spm_prediction
        
    @property
    def smm(self) -> SMMPrediction:
        if self.smm_prediction is not None:
            return self.smm_prediction
        else:
            return self._spm_to_smm()
        
    @property
    def sahi(self) -> SAHIPrediction:
        if self.sahi_prediction is not None:
            return self.sahi_prediction
        else:
            return self._spm_to_sahi()  
        
    def _smm_to_spm(self) -> SPMPrediction:
        """Convert SMMPrediction to SPMPrediction."""
        if self.smm_prediction is None:
            raise ValueError("No SMMPrediction available for conversion.")
        
        self.spm_prediction = SPMPrediction(
            names=[ann.type for ann in self.smm_prediction.annotations],
            segmentations=[ann.segmentation for ann in self.smm_prediction.annotations],
            polygons=[xy_mask_to_polygon([ann.segmentation]) for ann in self.smm_prediction.annotations],
            bboxes=[ann.bbox for ann in self.smm_prediction.annotations],
            class_ids=[ann.class_id for ann in self.smm_prediction.annotations],
            confidences=[ann.confidence for ann in self.smm_prediction.annotations],
            image_path=self.smm_prediction.image_name
        )
        return self.spm_prediction
    
    def _sahi_to_spm(self) -> SPMPrediction:
        """Convert SAHIPrediction to SPMPrediction."""
        if self.sahi_prediction is None:
            raise ValueError("No SAHIPrediction available for conversion.")
        
        self.spm_prediction = SPMPrediction(
            names=[pred.category.name for pred in self.sahi_prediction.predictions],
            segmentations=[pred.mask for pred in self.sahi_prediction.predictions],
            polygons=[xy_mask_to_polygon([pred.mask]) for pred in self.sahi_prediction.predictions],
            bboxes=[pred.bbox.to_xyxy() for pred in self.sahi_prediction.predictions],
            class_ids=[pred.category.id for pred in self.sahi_prediction.predictions],
            confidences=[pred.score.value for pred in self.sahi_prediction.predictions],
            image_path=None  # SAHI predictions may not have an associated image path
        )
        return self.spm_prediction
    
    def _spm_to_smm(self) -> SMMPrediction:
        """Convert SPMPrediction to SMMPrediction."""
        if self.spm_prediction is None:
            raise ValueError("No SPMPrediction available for conversion.")
        
        self.smm_prediction = SMMPrediction(image_name=self.spm_prediction.image_path)
        for name, segmentation, bbox, class_id, confidence in zip(
            self.spm_prediction.names,
            self.spm_prediction.segmentations,
            self.spm_prediction.bboxes,
            self.spm_prediction.class_ids,
            self.spm_prediction.confidences
        ):
            self.smm_prediction.add_annotation(
                type=name,
                class_id=class_id,
                confidence=confidence,
                bbox=bbox,
                segmentation=segmentation
            )
        return self.smm_prediction
    
    def _spm_to_sahi(self) -> SAHIPrediction:
        """Convert SPMPrediction to SAHIPrediction."""
        if self.spm_prediction is None:
            raise ValueError("No SPMPrediction available for conversion.")
        
        self.sahi_prediction = SAHIPrediction()
        for name, segmentation, bbox, class_id, confidence in zip(
            self.spm_prediction.names,
            self.spm_prediction.segmentations,
            self.spm_prediction.bboxes,
            self.spm_prediction.class_ids,
            self.spm_prediction.confidences
        ):
            sahi_segmentation = [s.flatten() for s in segmentation]
            # Create an ObjectPrediction instance for each annotation
            obj_pred = ObjectPrediction(
                category_name=name,
                category_id=class_id,
                score=confidence,
                bbox=bbox,
                segmentation=sahi_segmentation,
                full_shape=self.spm_prediction.image_shape
            )
            self.sahi_prediction.predictions.append(obj_pred)
        return self.sahi_prediction

@dataclass
class SPMPrediction:
    """Container for predictions from the model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""
    names: list[str] = field(default_factory=list)
    segmentations: list[list[list[tuple[float, float]]
                        ]] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]
                 ] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    image_path: Path = None
    image_shape: tuple[int, int] = None  # (height, width)

    def add_annotation(self, name: str = None, class_id: int = None, confidence: float = None, bbox: tuple[float] = None, segmentation: list[list[tuple[float]]] = None, polygon: Polygon = None):
        if polygon is None:
            # Assuming segmentation is a list of lists of (x, y) tuples
            polygon = self._to_polygon(segmentation)
        # TODO: Having both segmentation and polygon is redundant, we should consider only keeping one of them to save memory.
        self.polygons.append(polygon)
        self.segmentations.append(segmentation)
        # TODO: Bounding box size is likely to change after polygon simplification, so we may want to recalculate the bbox based on the final polygon.
        self.bboxes.append(polygon.bounds)
        self.class_ids.append(class_id)
        self.confidences.append(confidence)
        self.names.append(name)

    def _to_polygon(self, mask: list[list[tuple[float]]]) -> Polygon:
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
    
    def _get_crs_and_transform(self):
        if self.image_path is None:
            raise ValueError("image_path is not set for SPMPrediction.")
        with rasterio.open(self.image_path) as src:
            return src.crs, src.transform

    def _to_gdf(self) -> gpd.GeoDataFrame:
        """Convert the SPMPrediction to a GeoPandas GeoDataFrame."""
        from shapely.ops import transform
        
        crs, affine_transform = self._get_crs_and_transform()

        def pixel_to_geo(x, y, z=None):
            x_geo, y_geo = affine_transform * (x, y)
            return x_geo, y_geo

        geometries = [
            transform(pixel_to_geo, geom)
            for geom in self.polygons
        ]

        properties = {
            "name": self.names,
            "class_id": self.class_ids,
            "confidence": self.confidences,
            "bbox": self.bboxes,
        }

        gdf = gpd.GeoDataFrame(
            properties,
            geometry=geometries,
            crs=crs
        )

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

        crs, transform = self._get_crs_and_transform()
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

    def save_to_file(self, format: str = "geojson", name: str = None) -> None:
        """Save the SPMPrediction as a GeoJSON file."""
        format = format.lower()
        if format not in ["geojson", "gpkg"]:
            raise ValueError(f"Unsupported format: {format}")
        
        output_path = Path(f"{name}.{format}") if name else self.image_path.with_suffix(
                f".{format}") if self.image_path else Path(f"prediction.{format}")
        
        if format == "geojson":
            geojson_dict = self._to_geojson()
            import json
            with open(output_path, "w") as f:
                json.dump(geojson_dict, f, indent=2)

        elif format == "gpkg":
            gdf = self._to_gdf()
            gdf.to_file(output_path, driver="GPKG")

    @property
    def count(self):
        """Returns the number of annotations."""
        return len(self.class_ids)


@dataclass
class SPMConfig:
    """Configuration for the Spatial Polygon Merging (SPM) algorithm."""
    tau_dist: float = 3.0  # Distance around geometry within which to query the STRtree for neighbors
    tau_chain: int = 5  # Heuristic threshold to prevent excessive chaining of merges
