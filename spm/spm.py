from __future__ import annotations
from shapely import MultiPolygon, Polygon, STRtree, make_valid
from spm.config import SPMConfig, SPMPrediction
from utils.helpers import size_it, time_it
from utils.logger import logger


class SpatialPolygonMerger:
    def __init__(self, config: SPMConfig = SPMConfig()):
        self.config = config
        self.tree: STRtree = None
        self.annotations: SPMPrediction = None

    def index(self, annotations: SPMPrediction):
        """Creates a spatial index (STRtree) for the given list of polygons."""
        logger.info("Indexing polygons for merging")
        self.annotations = annotations
        self.tree = STRtree(self.annotations.polygons)

    def query(self, polygon: Polygon) -> list[int]:
        """Queries the spatial index for polygons that intersect with the given polygon."""
        if self.tree is None:
            raise ValueError("Spatial index not created. Call index() first.")
        return self.tree.query(polygon, predicate='dwithin', distance=self.config.tau_dist).tolist()
    
    def should_merge(self, poly_i, poly_j):
        """
        Merges two polygons that intersect or are within the distance threshold.
        """
        a = self.annotations.polygons[poly_i]
        b = self.annotations.polygons[poly_j]
        
        if a.intersects(b):
            return True
        
        return a.distance(b) <= self.config.tau_dist
    
    def merge_polygons(self, polygons: list[Polygon], gap_fill_distance: float = 5.0) -> tuple[Polygon, list[float, float, float, float]]:
        """
        Merge a cluster of fragment polygons into one.
        unary_union handles arbitrary numbers of fragments,
        including non-touching ones (produces MultiPolygon if disconnected,
        Polygon if contiguous).
        """
        from shapely.ops import unary_union
        cleaned = []
        for p in polygons:
            if p.is_empty:
                continue
            if not p.is_valid:
                p = make_valid(p)
            # Flatten any MultiPolygons from clipping
            if isinstance(p, MultiPolygon):
                cleaned.extend(p.geoms)
            else:
                cleaned.append(p)

        merged = unary_union(cleaned)
        # If fragments don't quite touch, optionally buffer slightly to close gaps
        if gap_fill_distance > 0:
            merged = merged.buffer(gap_fill_distance).buffer(-gap_fill_distance)
        # Calculate bounding box coordinates for the merged polygon
        minx, miny, maxx, maxy = merged.bounds
        return merged, [minx, miny, maxx, maxy]
    
    def find_neighbors(self, poly_idx: int) -> list[int]:
        """
        Find all chainable neighbors of a polygon based on distance and IoU thresholds.
        Args:
            poly_idx (int): Index of the polygon in self.annotations.polygons for which to find neighbors.
        Returns:
            list[int]: List of indices of neighboring polygons that should be merged with the given polygon.
            """
        polygon = self.annotations.polygons[poly_idx]
        neighbors = self.query(polygon)
        neighbors_to_merge = set([poly_idx])  # Start with the original polygon
        for neighbor_idx in neighbors:
            # Get the neighbors to merge for this neighbor
            if self.should_merge(poly_idx, neighbor_idx):
                neighbors_to_merge.add(neighbor_idx)
        while neighbors:
            neighbor_idx = neighbors.pop()
            query_polygon = self.annotations.polygons[neighbor_idx]
            sibling_neighbors = self.query(query_polygon)
            new_neighbors = set(sibling_neighbors) - neighbors_to_merge
            for new_neighbor_idx in new_neighbors:
                if self.should_merge(neighbor_idx, new_neighbor_idx):
                    neighbors_to_merge.add(new_neighbor_idx)
                    neighbors.append(new_neighbor_idx)

        return list(neighbors_to_merge)

    @size_it
    @time_it
    def merge(self, poly_idxs: list[int]=None) -> tuple[SPMPrediction]:
        """
        Merges polygons based on distance thresholds.
        Args:
            poly_idxs (list[int], optional): Optional list of polygon indices to consider for merging. If None, considers all polygons.
        Returns:
            tuple[SPMPrediction]: A tuple containing the merged annotations and the unmerged annotations.
        """
        if self.annotations is None:
            raise ValueError("No polygons to merge. Call index() first.")
        
        merged_annotations = SPMPrediction()
        unmerged_annotations = SPMPrediction()
        if poly_idxs is None:
            poly_idxs = list(range(len(self.annotations.polygons)))
        polygon_index = set(poly_idxs)  # Keep track of original indices
        undermerged_idxs = list(range(len(self.annotations.polygons)))

        while polygon_index:
            idx = polygon_index.pop()
            undermerged_idxs.remove(idx)
            neighbors = self.find_neighbors(idx)
            if idx in neighbors:
                neighbors.remove(idx)  # Remove self from neighbors
            polygons_to_merge = [idx]
            avg_score = self.annotations.confidences[idx]

            for neighbor_idx in neighbors:
                if (
                    neighbor_idx in polygon_index and 
                    self.annotations.class_ids[idx] == self.annotations.class_ids[neighbor_idx]
                    ):
                    polygons_to_merge.append(neighbor_idx)
                    avg_score = (avg_score + self.annotations.confidences[neighbor_idx]) / 2
                    polygon_index.remove(neighbor_idx)
                    undermerged_idxs.remove(neighbor_idx)
            merged_polygon, bbox = self.merge_polygons([self.annotations.polygons[i] for i in polygons_to_merge])
            if not merged_polygon:
                continue  # Skip if we couldn't merge into a single polygon
            merged_annotations.add_annotation(
                polygon=merged_polygon, 
                class_id=self.annotations.class_ids[idx],
                name=self.annotations.names[idx],
                confidence=avg_score, 
                bbox=bbox)
        
        # Get unmerged annotations that were not part of any merge
        for idx in undermerged_idxs:
            unmerged_annotations.add_annotation(
                polygon=self.annotations.polygons[idx],
                class_id=self.annotations.class_ids[idx],
                name=self.annotations.names[idx],
                confidence=self.annotations.confidences[idx],
                bbox=self.annotations.bboxes[idx] 
            )

        combined_annotations = SPMPrediction(
            polygons=merged_annotations.polygons + unmerged_annotations.polygons,
            class_ids=merged_annotations.class_ids + unmerged_annotations.class_ids,
            confidences=merged_annotations.confidences + unmerged_annotations.confidences,
            bboxes=merged_annotations.bboxes + unmerged_annotations.bboxes,
            names=merged_annotations.names + unmerged_annotations.names
        )
        
        return combined_annotations
