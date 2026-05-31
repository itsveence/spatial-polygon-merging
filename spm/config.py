from __future__ import annotations
from dataclasses import dataclass, field

from shapely import Polygon

from spm.utils import xy_mask_to_polygon

@dataclass
class SPMPrediction:
    """Container for predictions from the model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""
    names: list[str] = field(default_factory=list)
    segmentations: list[list[tuple[float, float]]] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def add_annotation(self, name: str = None, class_id: int = None, confidence: float = None, bbox: tuple[float, float, float, float] = None, segmentation: list[tuple[float, float]] = None, polygon: Polygon = None):
        if polygon is None:
            polygon = self._to_polygon(segmentation)  # Assuming segmentation is a list of lists of (x, y) tuples
        self.polygons.append(polygon)
        self.segmentations.append(segmentation)
        self.bboxes.append(bbox)
        self.class_ids.append(class_id)
        self.confidences.append(confidence)
        self.names.append(name)

    def _to_polygon(self, mask: list[tuple[float, float]]) -> Polygon:
        return xy_mask_to_polygon(mask)
    

@dataclass
class SPMConfig:
    """Configuration for the Spatial Polygon Merging (SPM) algorithm."""
    tau_dist: float = 10.0 # Distance around geometry within which to query the STRtree for neighbors
    rho_dist: float = 5.0 # Distance threshold for merging polygons (used in should_merge)