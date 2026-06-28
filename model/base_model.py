from abc import ABC, abstractmethod
from pathlib import Path

from spm.prediction import SPMPrediction

class BaseModel(ABC):
    """Abstract base class for all models used in the inference pipeline."""
    
    @abstractmethod
    def predict(self, image: Path) -> SPMPrediction:
        pass

    @abstractmethod
    def train(self, dataset: any) -> None:
        pass

