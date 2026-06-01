from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_path: str = field(default_factory=str)
    device: str = "cuda"  # or "cpu"
    batch_size: int = 4
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5

    # Inference parameters
    tile_size: int = 3000
    overlap: float = 0.1  # fraction of tile_size

    @property
    def overlap_pixels(self) -> int:
        return round(self.overlap * self.tile_size)
    
    contour_approx_factor: float = 0.01 # Factor for approximating contours