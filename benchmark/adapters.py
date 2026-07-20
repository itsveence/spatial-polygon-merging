from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from sahi.prediction import ObjectPrediction
from spatial_mask_merging.smm.predictions import SMMPrediction

from spm.core.geometry import xy_mask_to_polygon
from spm.core.prediction import SPMPrediction


@dataclass
class SAHIPrediction:
    """Container for predictions from the SAHI model, including masks, polygons, bounding boxes, class IDs, and confidence scores."""

    predictions: list[ObjectPrediction] = field(default_factory=list)


class PredictionAdapter:
    """Adapter class to handle conversion between different prediction objects [SPMPrediction, SMMPrediction, SAHIPrediction]"""

    def __init__(self, prediction: Union[SPMPrediction, SMMPrediction, SAHIPrediction]):
        assert isinstance(
            prediction, (SPMPrediction, SMMPrediction, SAHIPrediction)
        ), "prediction must be an instance of SPMPrediction, SMMPrediction, or SAHIPrediction"

        self.spm_prediction = (
            prediction if isinstance(prediction, SPMPrediction) else None
        )
        self.smm_prediction = (
            prediction if isinstance(prediction, SMMPrediction) else None
        )
        self.sahi_prediction = (
            prediction if isinstance(prediction, SAHIPrediction) else None
        )

        # If the input is SMMPrediction or SAHIPrediction, convert it to SPMPrediction
        if isinstance(prediction, SMMPrediction):
            self.spm_prediction = self._smm_to_spm()
        elif isinstance(prediction, SAHIPrediction):
            self.spm_prediction = self._sahi_to_spm()

    @property
    def spm(self) -> SPMPrediction:
        if self.spm_prediction is None:
            raise ValueError("No SPMPrediction available.")
        return self.spm_prediction

    @property
    def smm(self) -> SMMPrediction:
        if self.smm_prediction is not None:
            return self.smm_prediction
        else:
            return self._spm_to_smm()

    @property
    def sahi(self) -> SAHIPrediction:
        if self.sahi_prediction is not None:
            return self.sahi_prediction
        else:
            return self._spm_to_sahi()

    def _smm_to_spm(self) -> SPMPrediction:
        """Convert SMMPrediction to SPMPrediction."""
        if self.smm_prediction is None:
            raise ValueError("No SMMPrediction available for conversion.")

        self.spm_prediction = SPMPrediction(
            names=[ann.type for ann in self.smm_prediction.annotations],
            segmentations=[ann.segmentation for ann in self.smm_prediction.annotations],
            polygons=[
                xy_mask_to_polygon(ann.segmentation)
                for ann in self.smm_prediction.annotations
            ],
            bboxes=[ann.bbox for ann in self.smm_prediction.annotations],
            class_ids=[ann.class_id for ann in self.smm_prediction.annotations],
            confidences=[ann.confidence for ann in self.smm_prediction.annotations],
            image_path=Path(self.smm_prediction.image_name)
            if self.smm_prediction.image_name is not None
            else None,
        )
        return self.spm_prediction

    def _sahi_to_spm(self) -> SPMPrediction:
        """Convert SAHIPrediction to SPMPrediction."""
        if self.sahi_prediction is None:
            raise ValueError("No SAHIPrediction available for conversion.")

        self.spm_prediction = SPMPrediction(
            names=[pred.category.name for pred in self.sahi_prediction.predictions],
            segmentations=[pred.mask for pred in self.sahi_prediction.predictions],
            polygons=[
                xy_mask_to_polygon(
                    [np.array(seg).reshape(-1, 2) for seg in pred.mask.segmentation]
                )
                for pred in self.sahi_prediction.predictions
            ],
            bboxes=[pred.bbox.to_xyxy() for pred in self.sahi_prediction.predictions],
            class_ids=[pred.category.id for pred in self.sahi_prediction.predictions],
            confidences=[pred.score.value for pred in self.sahi_prediction.predictions],
            image_path=None,  # SAHI predictions may not have an associated image path
        )
        return self.spm_prediction

    def _spm_to_smm(self) -> SMMPrediction:
        """Convert SPMPrediction to SMMPrediction."""
        if self.spm_prediction is None:
            raise ValueError("No SPMPrediction available for conversion.")

        self.smm_prediction = SMMPrediction(
            image_name=str(self.spm_prediction.image_path)
            if self.spm_prediction.image_path is not None
            else ""
        )
        for name, segmentation, bbox, class_id, confidence in zip(
            self.spm_prediction.names,
            self.spm_prediction.segmentations,
            self.spm_prediction.bboxes,
            self.spm_prediction.class_ids,
            self.spm_prediction.confidences,
        ):
            self.smm_prediction.add_annotation(
                type=name,
                class_id=class_id,
                confidence=confidence,
                bbox=bbox,
                segmentation=segmentation,
            )
        return self.smm_prediction

    def _spm_to_sahi(self) -> SAHIPrediction:
        """Convert SPMPrediction to SAHIPrediction."""
        if self.spm_prediction is None:
            raise ValueError("No SPMPrediction available for conversion.")

        self.sahi_prediction = SAHIPrediction()
        for name, segmentation, bbox, class_id, confidence in zip(
            self.spm_prediction.names,
            self.spm_prediction.segmentations,
            self.spm_prediction.bboxes,
            self.spm_prediction.class_ids,
            self.spm_prediction.confidences,
        ):
            sahi_segmentation = [np.asarray(s).flatten() for s in segmentation]
            # Create an ObjectPrediction instance for each annotation
            obj_pred = ObjectPrediction(
                category_name=name,
                category_id=class_id,
                score=confidence,
                bbox=bbox,
                segmentation=sahi_segmentation,
                full_shape=self.spm_prediction.image_shape,
            )
            self.sahi_prediction.predictions.append(obj_pred)
        return self.sahi_prediction
