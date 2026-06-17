from __future__ import annotations
from collections import deque
from shapely import MultiPolygon, Polygon, STRtree, make_valid
from spm.config import SPMConfig, SPMPrediction
from utils.helpers import size_it, time_it
from utils.logger import logger


class SpatialPolygonMerger:
    def __init__(self, config: SPMConfig = SPMConfig()):
        self.config = config
        self.tree: STRtree = None
        self.annotations: SPMPrediction = None
        self.query_cache: dict[int, list[int]] = {}
        
    @time_it
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
        
        return a.distance(b) <= self.config.rho_dist
    
    def merge_polygons(self, polygons: list[Polygon]) -> tuple[Polygon, list[float, float, float, float]]:
        """
        Merge a cluster of fragment polygons into one.
        unary_union handles arbitrary numbers of fragments,
        including non-touching ones (produces MultiPolygon if disconnected,
        Polygon if contiguous).
        """
        from shapely.ops import unary_union

        merged = unary_union(polygons)

        merged = merged.buffer(0)  # Clean up geometry

        # Calculate bounding box coordinates for the merged polygon
        minx, miny, maxx, maxy = merged.bounds
        return merged, [minx, miny, maxx, maxy]
    
    def _query(self, poly_idx: int) -> list[int]:
        """
        Internal method to query neighbors for a given polygon index with caching.
        """
        if cached:= self.query_cache.get(poly_idx):
            return cached
        
        neighbors = self.query(self.annotations.polygons[poly_idx])
        self.query_cache[poly_idx] = neighbors
        return neighbors
    
    def find_neighbors(self, poly_idx: int) -> set[int]:
        """
        Find all chainable neighbors of a polygon with BFS.
        Args:
            poly_idx (int): Index of the polygon in self.annotations.polygons for which to find neighbors.
        Returns:
            set[int]: Set of indices of neighboring polygons that should be merged with the given polygon.
            """
        
        # anchor = self.annotations.polygons[poly_idx]
        neighbors_to_merge = {poly_idx}
        queue: deque[tuple[int, int]] = deque()

        # Base case: find direct neighbors of the anchor polygon
        for cand_idx in self._query(poly_idx):
            if cand_idx == poly_idx:
                continue
            # if self.should_merge(poly_idx, cand_idx):
            neighbors_to_merge.add(cand_idx)
            queue.append((cand_idx, 1))

        # Transitive: each new merge increments depth from its parent
        while queue:
            node_idx, depth = queue.popleft()
            if depth >= self.config.tau_chain:
                continue
            for cand_idx in self._query(node_idx):
                if cand_idx in neighbors_to_merge:
                    continue
                # if self.should_merge(node_idx, cand_idx):
                neighbors_to_merge.add(cand_idx)
                queue.append((cand_idx, depth + 1))

        return neighbors_to_merge

    @size_it
    @time_it
    def merge(self, poly_idxs: list[int]=None) -> SPMPrediction:
        """
        Merges polygons based on distance thresholds.
        Args:
            poly_idxs (list[int], optional): Optional list of polygon indices to consider for merging. If None, considers all polygons.
        Returns:
            SPMPrediction: The merged annotations.
        """
        if self.annotations is None:
            raise ValueError("No polygons to merge. Call index() first.")
        
        merged_annotations = SPMPrediction(image_path=self.annotations.image_path)
        unmerged_annotations = SPMPrediction(image_path=self.annotations.image_path)
        if poly_idxs is None:
            poly_idxs = list(range(len(self.annotations.polygons)))
        polygon_index = set(poly_idxs)  # Keep track of original indices
        undermerged_idxs = set(range(len(self.annotations.polygons)))

        while polygon_index:
            idx = polygon_index.pop()
            undermerged_idxs.discard(idx)
            neighbors = self.find_neighbors(idx)
            logger.debug(f"Polygon {idx} has neighbors to merge: {neighbors}")
            if idx in neighbors:
                neighbors.discard(idx)
            polygons_to_merge = [idx]
            score_list = [self.annotations.confidences[idx]]

            for neighbor_idx in neighbors:
                if (
                    neighbor_idx in polygon_index and 
                    self.annotations.class_ids[idx] == self.annotations.class_ids[neighbor_idx]
                    ):
                    polygons_to_merge.append(neighbor_idx)
                    score_list.append(self.annotations.confidences[neighbor_idx])
                    polygon_index.discard(neighbor_idx)
                    undermerged_idxs.discard(neighbor_idx)

            avg_score = sum(score_list) / len(score_list)
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

        logger.info(f"Total merged polygons: {len(merged_annotations.polygons)}")
        logger.info(f"Total unmerged polygons: {len(unmerged_annotations.polygons)}")
        logger.info(f"Total polygons after merging: {len(merged_annotations.polygons) + len(unmerged_annotations.polygons)}")

        combined_annotations = SPMPrediction(
            polygons=merged_annotations.polygons + unmerged_annotations.polygons,
            class_ids=merged_annotations.class_ids + unmerged_annotations.class_ids,
            confidences=merged_annotations.confidences + unmerged_annotations.confidences,
            bboxes=merged_annotations.bboxes + unmerged_annotations.bboxes,
            names=merged_annotations.names + unmerged_annotations.names,
            image_path=self.annotations.image_path
        )
        
        return combined_annotations

