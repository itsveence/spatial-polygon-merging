from __future__ import annotations
from dataclasses import dataclass
from shapely import Polygon, STRtree
from spm.predictions import SPMPrediction
from utils.helpers import time_it

@dataclass
class SPMConfig:
    """Configuration for the Spatial Polygon Merging (SPM) algorithm."""
    tau_iou: float = 0.5
    tau_dist: float = 10.0

class SpatialPolygonMerger:
    def __init__(self, config: SPMConfig = SPMConfig()):
        self.config = config
        self.tree: STRtree = None
        self.annotations: SPMPrediction = None

    def index(self, annotations: SPMPrediction):
        """Creates a spatial index (STRtree) for the given list of polygons."""
        self.annotations = annotations
        self.tree = STRtree(self.annotations.polygons)

    def query(self, polygon: Polygon) -> list[int]:
        """Queries the spatial index for polygons that intersect with the given polygon."""
        if self.tree is None:
            raise ValueError("Spatial index not created. Call index() first.")
        return self.tree.query(polygon, predicate='dwithin', distance=self.config.tau_dist).tolist()
    
    def should_merge(self, poly_i, poly_j):
        """
        Merge criterion for cross-tile fragment pairs.
        Either close enough (boundary distance) or overlapping enough (IoU).
        """
        a = self.annotations.polygons[poly_i]
        b = self.annotations.polygons[poly_j]
        dist = a.distance(b)
        if dist > self.config.tau_dist:
            return False
        if dist == 0.0:  # overlapping or touching — check IoU
            intersection = a.intersection(b).area
            union = a.union(b).area
            iou = intersection / union if union > 0 else 0.0
            return iou >= self.config.tau_iou
        return True  # within tau_d and non-overlapping — merge
    
    def merge_polygons(self, polygons: list[Polygon], gap_fill_distance: float = 5.0) -> tuple[Polygon, list[float, float, float, float]]:
        """
        Merge a cluster of fragment polygons into one.
        unary_union handles arbitrary numbers of fragments,
        including non-touching ones (produces MultiPolygon if disconnected,
        Polygon if contiguous).
        """
        from shapely.ops import unary_union
        merged = unary_union(polygons)
        # If fragments don't quite touch, optionally buffer slightly to close gaps
        if merged.geom_type == 'MultiPolygon':
            merged = merged.buffer(gap_fill_distance).buffer(-gap_fill_distance)
        # Calculate bounding box coordinates for the merged polygon
        minx, miny, maxx, maxy = merged.bounds
        return merged, [minx, miny, maxx, maxy]

    @time_it
    def merge(self) -> SPMPrediction:
        """Merges polygons based on IoU and distance thresholds."""
        if self.annotations is None:
            raise ValueError("No polygons to merge. Call index() first.")
        
        merged_annotations = SPMPrediction()
        polygon_index = set(i for i in range(len(self.annotations.polygons)))  # Keep track of original indices
        while polygon_index:
            idx = polygon_index.pop()
            polygon = self.annotations.polygons[idx]
            neighbors = self.query(polygon)
            neighbors.remove(idx)  # Remove self from neighbors
            polygons_to_merge = [idx]
            avg_score = self.annotations.confidences[idx]

            for neighbor_idx in neighbors:
                if (
                    neighbor_idx in polygon_index and 
                    self.annotations.class_ids[idx] == self.annotations.class_ids[neighbor_idx] and 
                    self.should_merge(idx, neighbor_idx)
                    ):
                    polygons_to_merge.append(neighbor_idx)
                    avg_score = (avg_score + self.annotations.confidences[neighbor_idx]) / 2
                    polygon_index.remove(neighbor_idx)
            merged_polygon, bbox = self.merge_polygons([self.annotations.polygons[i] for i in polygons_to_merge])
            merged_annotations.add_annotation(polygon=merged_polygon, class_id=self.annotations.class_ids[idx], confidence=avg_score, bbox=bbox)

        
        return merged_annotations
