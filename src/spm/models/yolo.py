import cv2
import numpy as np
import rasterio
import torch
from ultralytics import YOLO
from pathlib import Path

from spm.core.prediction import SPMPrediction
from spm.merging.spm import SpatialPolygonMerger
from spm.config import ModelConfig
from spm.visualization.overlays import visualize
from spm.models.base_model import BaseModel
from spm.preprocessing.image_processing import binary_mask_to_contours, stream_tiles_by_batch
from spm.core.geometry import effective_overlap, is_overlap_candidate
from spm.utils.logger import logger


class YOLOModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.model.eval()  # Set model to evaluation mode

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _process_binary_mask(self, mask: np.ndarray) -> np.ndarray | None:
        contours = binary_mask_to_contours(
            mask, min_area=5, contour_approx_factor=self.config.contour_approx_factor, normalize=False, sort_by_area=True)

        if len(contours) == 0:
            return None

        # Get the largest contour as the main polygon
        contour = contours[0]

        # Scale polygon points back to original image coordinates
        contour = np.asarray(contour, dtype=np.float32)
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)

        # Scale polygon points back to original image size
        scale_x = self.config.tile_size / mask.shape[1]
        scale_y = self.config.tile_size / mask.shape[0]

        processed_mask = contour * \
            np.array([scale_x, scale_y], dtype=np.float32)

        return [processed_mask]

    def predict(
            self,
            image: Path,
            merge: bool = True,
            merge_only_border: bool = True,
            get_seg_from_binary_mask: bool = False
            ) -> SPMPrediction:
        """Generates prediction for the given image and returns it as an SPMPrediction object.

        Args:
            image (Path): Path to the input image for prediction.
            merge (bool): Whether to merge overlapping predictions using SPM.
            merge_only_border (bool): Whether to only merge border predictions (Ignored if merge is False).
            get_seg_from_binary_mask (bool): Whether to get segmentation from a binary mask.

        Returns:
            SPMPrediction: The merged prediction result.
        """
        if isinstance(image, str):
            image = Path(image)

        prediction = SPMPrediction(image_path=image)

        border_prediction_idxs: list[int] = []

        with rasterio.open(image) as src:

            # Get image height and width for effective overlap calculation
            img_width, img_height = src.width, src.height
            prediction.image_shape = (img_height, img_width)  # Store image shape in prediction
            logger.info(
                f"Performing prediction on image: {image} (width: {img_width}, height: {img_height})")

            prediction_idx = 0
            tile_counter = 0
            # Stream tiles from the image and run inference on each tile
            for batch, coordinate in stream_tiles_by_batch(src, tile_size=self.config.tile_size, batch_size=self.config.batch_size, overlap=self.config.overlap_pixels):
                tile_counter += len(batch)
                with torch.inference_mode():
                    results = self.model(
                        batch,
                        device=self.config.device,
                        conf=self.config.confidence_threshold,
                        iou=self.config.iou_threshold,
                        verbose=False
                    )

                for i, res in enumerate(results):
                    if res.boxes.xyxy.shape[0] == 0:
                        logger.debug(
                            f"No boxes found in tile {i} at coordinate {coordinate[i]}")
                        continue

                    tile_overlap = effective_overlap(
                        tile_x=coordinate[i][1],
                        tile_y=coordinate[i][0],
                        tile_size=self.config.tile_size,
                        overlap=self.config.overlap_pixels,
                        img_width=img_width,
                        img_height=img_height,
                    )

                    tile_x, tile_y = coordinate[i][1], coordinate[i][0]

                    bboxes = res.boxes.xyxy.cpu().numpy()      # (N, 4)
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    confidences = res.boxes.conf.cpu().numpy()
                    names = res.names
                    masks = (
                        res.masks.data.cpu().numpy().astype(np.uint8)
                        if get_seg_from_binary_mask else res.masks.xy
                    )

                    # Convert tile-relative coordinates to absolute image coordinates
                    abs_bboxes = bboxes.copy()
                    abs_bboxes[:, [0, 2]] += tile_x
                    abs_bboxes[:, [1, 3]] += tile_y

                    for j in range(len(bboxes)):
                        if len(bboxes[j]) != 4 or len(masks[j]) < 4:
                            logger.debug(
                                f"No boxes found in tile {i} at coordinate {coordinate[i]}")
                            continue

                        _cls = int(classes[j])
                        _conf = float(confidences[j])
                        _bbox = abs_bboxes[j]
                        _mask = masks[j]

                        if get_seg_from_binary_mask:

                            # Convert binary mask to polygon
                            _mask = self._process_binary_mask(_mask)

                            # Sometimes the binary mask may not yield valid contours, so we skip those cases
                            if _mask is None:
                                logger.debug(
                                    f"No valid contours found in binary mask for tile {i} at coordinate {coordinate[i]}")
                                continue
                        
                        for seg_idx in range(len(_mask)):

                            # Adjust the mask coordinates to the original image coordinate system
                            _mask[seg_idx][:, 0] = (_mask[seg_idx][:, 0] + tile_x)
                            _mask[seg_idx][:, 1] = (_mask[seg_idx][:, 1] + tile_y)

                        if merge and is_overlap_candidate(
                                bbox=bboxes[j],
                                tile_size=self.config.tile_size,
                                overlap=tile_overlap,
                                tile_x=tile_x, tile_y=tile_y,
                                img_width=img_width,
                                img_height=img_height
                        ):
                            border_prediction_idxs.append(prediction_idx)

                        prediction.add_annotation(
                            name=names[_cls],
                            class_id=_cls,
                            confidence=_conf,
                            bbox=_bbox,
                            segmentation=_mask
                        )
                        prediction_idx += 1

            logger.info(f"Total tiles processed: {tile_counter}")

            if merge:
                # Merge border predictions using SPM
                smp = SpatialPolygonMerger()
                smp.index(prediction)

                if merge_only_border:
                    prediction = smp.merge(border_prediction_idxs)
                else:
                    prediction = smp.merge()

        return prediction

    def train(self, dataset):
        # Implement the training logic here
        pass
