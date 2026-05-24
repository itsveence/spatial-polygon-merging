from __future__ import annotations
from dataclasses import dataclass, field

from shapely import Polygon

from spm.utils import xy_mask_to_polygon

@dataclass
class SPMPrediction:
    """Container for predictions from the model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""
    masks: list[list[tuple[float, float]]] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def add_annotation(self, class_id: int = None, confidence: float = None, bbox: tuple[float, float, float, float] = None, mask: list[tuple[float, float]] = None, polygon: Polygon = None):
        if polygon is None:
            polygon = self._to_polygon(mask)
        self.polygons.append(polygon)
        self.masks.append(mask)
        self.bboxes.append(bbox)
        self.class_ids.append(class_id)
        self.confidences.append(confidence)

    def _to_polygon(self, mask: list[tuple[float, float]]) -> Polygon:
        return xy_mask_to_polygon(mask)