from __future__ import annotations

from collections import deque
from shapely import MultiPolygon, Polygon, STRtree

from spm.config.config import SPMConfig
from spm.core.prediction import SPMPrediction
from spm.utils.logger import logger
from typing import Optional


class SpatialPolygonMerger:
    def __init__(self, config: SPMConfig = SPMConfig()):
        self.config = config
        self.tree: Optional[STRtree] = None
        self.annotations: Optional[SPMPrediction] = None
        self.query_cache: dict[int, list[int]] = {}

    def index(self, annotations: SPMPrediction):
        """Creates a spatial index (STRtree) for the given list of polygons."""
        logger.info("Indexing polygons for merging")
        self.annotations = annotations
        self.tree = STRtree(self.annotations.polygons)

    def query(self, polygon: Polygon) -> list[int]:
        """Queries the spatial index for polygons that intersect with the given polygon."""
        if self.tree is None:
            raise ValueError("Spatial index not created. Call index() first.")
        return self.tree.query(
            polygon, predicate="dwithin", distance=self.config.tau_dist
        ).tolist()

    def should_merge(self, poly_i, poly_j):
        """
        Merges two polygons that intersect or are within the distance threshold.
        """
        a = self.annotations.polygons[poly_i]
        b = self.annotations.polygons[poly_j]

        if a.intersects(b):
            return True

        return a.distance(b) <= self.config.tau_dist

    @staticmethod
    def _segmentation_from_geometry(
        geometry: Polygon | MultiPolygon,
    ) -> list[list[tuple[float, float]]]:
        """Extract exterior-ring coordinates as a segmentation list."""
        if isinstance(geometry, MultiPolygon):
            return [list(part.exterior.coords) for part in geometry.geoms]
        return [list(geometry.exterior.coords)]

    def merge_polygons(self, polygons: list[Polygon]) -> Polygon:
        """
        Merge a cluster of fragment polygons into one.
        unary_union handles arbitrary numbers of fragments,
        including non-touching ones (produces MultiPolygon if disconnected,
        Polygon if contiguous).
        """
        from shapely.ops import unary_union

        logger.debug(f"Merging {len(polygons)} polygons")
        merged = unary_union(polygons)

        merged = merged.buffer(0)  # Clean up geometry

        return merged

    def _query(self, poly_idx: int) -> list[int]:
        """
        Internal method to query neighbors for a given polygon index with caching.
        """
        if cached := self.query_cache.get(poly_idx):
            return cached

        assert self.annotations is not None
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

        assert self.annotations is not None
        # anchor = self.annotations.polygons[poly_idx]
        neighbors_to_merge = {poly_idx}
        queue: deque[tuple[int, int]] = deque()

        # Base case: find direct neighbors of the anchor polygon
        for cand_idx in self._query(poly_idx):
            if cand_idx == poly_idx:
                continue
            if (
                self.annotations.class_ids[poly_idx]
                == self.annotations.class_ids[cand_idx]
            ):
                neighbors_to_merge.add(cand_idx)
                queue.append((cand_idx, 1))

        # Transitive: each new merge increments depth from its parent
        while queue:
            node_idx, depth = queue.popleft()
            if depth < self.config.rho_chain:
                for cand_idx in self._query(node_idx):
                    if cand_idx in neighbors_to_merge:
                        continue
                    if (
                        self.annotations.class_ids[poly_idx]
                        == self.annotations.class_ids[cand_idx]
                    ):
                        neighbors_to_merge.add(cand_idx)
                        queue.append((cand_idx, depth + 1))

        return neighbors_to_merge

    def merge(self, poly_idxs: Optional[list[int]] = None) -> SPMPrediction:
        """
        Merges polygons based on distance thresholds.
        Args:
            poly_idxs (list[int], optional): Optional list of polygon indices to consider for merging. If None, considers all polygons.
        Returns:
            SPMPrediction: The merged annotations.
        """
        if self.annotations is None:
            raise ValueError("No polygons to merge. Call index() first.")

        # logger.info(f"Starting merge process with {len(self.annotations.polygons)} polygons.")

        merged_annotations = SPMPrediction(
            image_path=self.annotations.image_path,
            image_shape=self.annotations.image_shape,
        )

        # If no specific polygon indices are provided, consider all polygons for merging
        if poly_idxs is None:
            poly_idxs = list(range(len(self.annotations.polygons)))

        polygon_index = set(poly_idxs)  # Keep track of original indices
        undermerged_idxs = set(range(len(self.annotations.polygons)))

        while polygon_index:
            idx = polygon_index.pop()
            neighbors = self.find_neighbors(idx)
            logger.debug(f"Polygon {idx} has neighbors to merge: {neighbors}")

            if len(neighbors) <= 1:
                continue

            neighbors.discard(idx)
            polygons_to_merge = [idx]
            score_sum = self.annotations.confidences[idx]
            score_count = 1

            for neighbor_idx in neighbors:
                if (
                    neighbor_idx in undermerged_idxs
                    and self.annotations.class_ids[idx]
                    == self.annotations.class_ids[neighbor_idx]
                ):
                    polygons_to_merge.append(neighbor_idx)

                    score_sum += self.annotations.confidences[neighbor_idx]
                    score_count += 1

                    polygon_index.discard(neighbor_idx)
                    undermerged_idxs.discard(neighbor_idx)

            if len(polygons_to_merge) == 1:
                continue

            merged_polygon = self.merge_polygons(
                [self.annotations.polygons[i] for i in polygons_to_merge]
            )
            if not merged_polygon:
                continue

            merged_annotations.add_annotation(
                polygon=merged_polygon,
                segmentation=self._segmentation_from_geometry(merged_polygon),
                class_id=self.annotations.class_ids[idx],
                name=self.annotations.names[idx],
                confidence=(score_sum / score_count),
                bbox=merged_polygon.bounds,
            )

            undermerged_idxs.discard(idx)

        # Get unmerged annotations that were not part of any merge
        while undermerged_idxs:
            idx = undermerged_idxs.pop()
            merged_annotations.add_annotation(
                polygon=self.annotations.polygons[idx],
                segmentation=self._segmentation_from_geometry(
                    self.annotations.polygons[idx]
                ),
                class_id=self.annotations.class_ids[idx],
                name=self.annotations.names[idx],
                confidence=self.annotations.confidences[idx],
                bbox=self.annotations.bboxes[idx],
            )
        merged_annotations.image_path = self.annotations.image_path
        merged_annotations.image_shape = self.annotations.image_shape

        logger.debug(
            f"Total polygons after merging: {len(merged_annotations.polygons)}"
        )

        return merged_annotations
