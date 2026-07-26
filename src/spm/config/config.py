from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_path: str = field(default_factory=str)
    device: str = "cuda"  # or "cpu"
    batch_size: int = 4
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5

    # Inference parameters
    tile_size: int = 1500
    overlap: float = 0.1  # fraction of tile_size

    @property
    def overlap_pixels(self) -> int:
        return round(self.overlap * self.tile_size)

    contour_approx_factor: float = 0.01  # Factor for approximating contours


ScoreAgg = Literal[
    "weighted_mean", "max", "mean", "min"
]  # Type for score aggregation methods


@dataclass
class SPMConfig:
    """Configuration for the Spatial Polygon Merging (SPM) algorithm."""

    tau_dist: float = (
        1.0  # Distance around geometry within which to query the STRtree for neighbors
    )
    rho_chain: int = 5  # Heuristic threshold to prevent excessive chaining of merges
    score_agg: ScoreAgg = (
        "weighted_mean"  # Method for aggregating scores of merged polygons
    )
