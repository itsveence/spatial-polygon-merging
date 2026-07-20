from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from spm.core.prediction import SPMPrediction


class BaseModel(ABC):
    """Abstract base class for all models used in the inference pipeline."""

    @abstractmethod
    def predict(self, image: Path) -> SPMPrediction:
        pass

    @abstractmethod
    def train(self, dataset: Any) -> None:
        pass
