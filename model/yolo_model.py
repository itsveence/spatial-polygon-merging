import cv2
import numpy as np
import rasterio
import torch
from ultralytics import YOLO
from pathlib import Path

from model.base_model import BaseModel
from model.config import ModelConfig
from preprocessing.image_processing import stream_tiles_by_batch
from spm.config import SPMPrediction
from spm.spm import SpatialPolygonMerger
from spm.utils import effective_overlap, is_overlap_candidate
from utils.helpers import size_it, time_it
from utils.logger import logger

class YOLOModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        # Load the model here using the provided model path and device
        self.model = YOLO(config.model_path)
        self.model.eval()  # Set model to evaluation mode

    def __call__(self, *args, **kwargs) -> SPMPrediction:
        """Performs prediction on the given image and returns the result as an SPMPrediction object."""
        prediction = self.predict(*args, **kwargs)
        return prediction

    @staticmethod
    def visualize(src: rasterio.DatasetReader, prediction: SPMPrediction, output_path: Path) -> None:
        """Visualizes the predictions on the original image and saves the output.

        Args:
            src (rasterio.DatasetReader): The original input image.
            prediction (SPMPrediction): The prediction result containing polygons, class IDs, confidences, and bounding boxes.
            output_path (Path): Path to save the visualized output image.
        """
        logger.info(f"Visualizing predictions and saving to {output_path}")

        # --- Style constants ---------------------------------------------------
        FILL_ALPHA = 0.4
        FILL_COLOR = (60, 200, 255)
        OUTLINE_COLOR = (60, 200, 255)
        BOX_COLOR = (80, 220, 120)
        TEXT_COLOR = (255, 255, 255)
        OUTLINE_THICKNESS = 2
        BOX_THICKNESS = 2
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.5
        FONT_THICKNESS = 1

        # --- Prepare base image ------------------------------------------------
        img = src.read().transpose(1, 2, 0)
        img_bgr = img[:, :, :3][..., ::-1]
        img_bgr = np.ascontiguousarray(img_bgr)

        # Overlay holds ONLY the segmentation fills, so blending affects fills alone.
        overlay = img_bgr.copy()

        def _fill(pts):
            cv2.fillPoly(overlay, [pts], color=FILL_COLOR, lineType=cv2.LINE_AA)

        # Collect geometry to draw crisp elements after the blend.
        outlines = []
        annotations = []

        for poly, bbox, confidence, name in zip(
            prediction.polygons, prediction.bboxes, prediction.confidences, prediction.names
        ):
            if poly.is_empty:
                continue

            sub_polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
            for subpoly in sub_polys:
                if subpoly.is_empty:
                    continue
                pts = np.array(subpoly.exterior.coords, dtype=np.int32)
                _fill(pts)
                outlines.append(pts)

            annotations.append((bbox, f"{name}: {confidence:.2f}"))

        # --- Blend the fills back (only the segmentation is transparent) -------
        cv2.addWeighted(overlay, FILL_ALPHA, img_bgr, 1 - FILL_ALPHA, 0, dst=img_bgr)

        # --- Draw crisp, fully-opaque elements on top -------------------------
        # Polygon outlines give the segmentation a defined edge.
        for pts in outlines:
            cv2.polylines(img_bgr, [pts], isClosed=True, color=OUTLINE_COLOR,
                        thickness=OUTLINE_THICKNESS, lineType=cv2.LINE_AA)

        # Bounding boxes + labels with a filled background for readability.
        for bbox, label in annotations:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS, lineType=cv2.LINE_AA)

            (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            # Keep the label inside the image if the box is near the top edge.
            label_y = y1 - 6 if y1 - th - 6 > 0 else y1 + th + 6
            bg_top = label_y - th - baseline
            cv2.rectangle(img_bgr, (x1, bg_top), (x1 + tw + 6, label_y + baseline),
                        BOX_COLOR, thickness=cv2.FILLED)
            cv2.putText(img_bgr, label, (x1 + 3, label_y), FONT, FONT_SCALE,
                        TEXT_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)

        cv2.imwrite(str(output_path), img_bgr)

    @size_it
    @time_it
    def predict(self, image: Path, save: bool = False, merge_only_border: bool = True) -> SPMPrediction:
        """Generates prediction for the given image and returns it as an SPMPrediction object.

        Args:
            image (Path): Path to the input image for prediction.
            save (bool): Whether to save the predictions.
            merge_only_border (bool): Whether to only merge border predictions.

        Returns:
            SPMPrediction: The prediction result containing polygons, class IDs, confidences, and bounding boxes.
        """
        if isinstance(image, str):
            image = Path(image)

        prediction = SPMPrediction()

        border_prediction_idxs: list[int] = []

        with rasterio.open(image) as src:
            # Get image height and width for effective overlap calculation
            img_width, img_height = src.width, src.height
            logger.info(f"Performing prediction on image: {image} (width: {img_width}, height: {img_height})")

            prediction_idx = 0
            tile_counter = 0
            # Stream tiles from the image and run inference on each tile
            for batch, coordinate in stream_tiles_by_batch(src, tile_size=self.config.tile_size, batch_size=self.config.batch_size, overlap=self.config.overlap):
                tile_counter += len(batch)
                with torch.inference_mode():
                    results = self.model(batch, device=self.config.device, conf=self.config.confidence_threshold, iou=self.config.iou_threshold, verbose=False)

                for i, res in enumerate(results):
                    tile_overlap = effective_overlap(
                    tile_x=coordinate[i][1],
                    tile_y=coordinate[i][0],
                    tile_size=self.config.tile_size,
                    overlap=self.config.overlap,
                    img_width=img_width,
                    img_height=img_height,
                )
                    if res.boxes.xyxy.shape[0] == 0:
                        logger.debug(f"No boxes found in tile {i} at coordinate {coordinate[i]}")
                        continue
                    for j, (bbox, mask, cls, conf) in enumerate(zip(res.boxes.xyxy, res.masks.xy, res.boxes.cls, res.boxes.conf)):
                        if len(bbox) != 4 or len(mask) < 4:
                            logger.debug(f"No boxes found in tile {i} at coordinate {coordinate[i]}")
                            continue
                        _bbox = bbox.detach().clone()
                        _mask = mask.copy()
                        _cls = int(cls.detach().clone())
                        _conf = float(conf.detach().clone())

                        # Correct the mask coordinates to the original image coordinate system
                        _mask[:, 0] += coordinate[i][1]  # Adjust x coordinates by left offset
                        _mask[:, 1] += coordinate[i][0]  # Adjust y coordinates by top offset
                        # Correct the bbox coordinates to the original image coordinate system
                        _bbox[0] = _bbox[0] + coordinate[i][1]  # x_min
                        _bbox[1] = _bbox[1] + coordinate[i][0]  # y_min
                        _bbox[2] = _bbox[2] + coordinate[i][1]  # x_max
                        _bbox[3] = _bbox[3] + coordinate[i][0]  # y_max

                        if is_overlap_candidate(bbox=bbox, tile_size=self.config.tile_size, overlap=tile_overlap, tile_x=coordinate[i][1], tile_y=coordinate[i][0], img_width=img_width, img_height=img_height):
                            border_prediction_idxs.append(prediction_idx)

                        prediction.add_annotation(
                            name=res.names[_cls],  # class label
                            class_id=_cls,
                            confidence=_conf,
                            bbox=_bbox,
                            segmentation=_mask
                        )
                        prediction_idx += 1
            logger.info(f"Total tiles processed: {tile_counter}")
            # Merge border predictions using SPM
            smp = SpatialPolygonMerger()
            smp.index(prediction)
            if merge_only_border:
                merged_polygons = smp.merge(border_prediction_idxs)
            else:
                merged_polygons = smp.merge()

            if save:
                output_path = image.parent / f"{image.stem}_prediction.png"
                self.visualize(src, merged_polygons, output_path)
        return merged_polygons

    def train(self, dataset):
        # Implement the training logic here
        pass


if __name__ == "__main__":
    config = ModelConfig(
        model_path="runs/segment/yolo-seg-whu/weights/best.pt",
        device="cuda:0",
        tile_size=1500,
        batch_size=4,
        overlap=300
    )
    model = YOLOModel(config)
    prediction = model(Path("christchurch_487.tif"), save=False, merge_only_border=True)