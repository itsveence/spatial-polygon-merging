import cv2
import numpy as np
import rasterio
from ultralytics import YOLO
from pathlib import Path

from model.base_model import BaseModel
from model.config import ModelConfig
from preprocessing.image_processing import stream_tiles_by_batch
from spm.config import SPMPrediction
from spm.spm import SpatialPolygonMerger
from spm.utils import effective_overlap, is_border_candidate
from utils.logger import logger

class YOLOModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        # Load the model here using the provided model path and device
        self.model = YOLO(config.model_path)

    @staticmethod
    def visualize(src: rasterio.DatasetReader, prediction: SPMPrediction, output_path: Path) -> None:
        """Visualizes the predictions on the original image and saves the output.

        Args:
            src (rasterio.DatasetReader): The original input image.
            prediction (SPMPrediction): The prediction result containing polygons, class IDs, confidences, and bounding boxes.
            output_path (Path): Path to save the visualized output image.
        """
        # Implement visualization logic here using libraries like OpenCV or Matplotlib
        img = src.read().transpose(1, 2, 0)
        img_bgr = img[:, :, :3][..., ::-1]
        img_bgr = np.ascontiguousarray(img_bgr)  # make contiguous for OpenCV
        overlay = img_bgr.copy()
        for poly, bbox, confidence, name in zip(prediction.polygons, prediction.bboxes, prediction.confidences, prediction.names):
            if poly.is_empty:
                continue
            if poly.geom_type == 'Polygon':
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                # Draw the polygon on the image
                cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
                # Draw bounding box
                cv2.rectangle(img_bgr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                # Put class label and confidence
                label = f"{name}: {confidence:.2f}"
                cv2.putText(img_bgr, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif poly.geom_type == 'MultiPolygon':
                for subpoly in poly.geoms:
                    pts = np.array(subpoly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
                    cv2.rectangle(img_bgr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    label = f"{name}: {confidence:.2f}"
                    cv2.putText(img_bgr, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.addWeighted(overlay, alpha=0.4, src2=img_bgr, beta=0.6, gamma=0, dst=img_bgr)
        cv2.imwrite(str(output_path), img_bgr)

    def predict(self, image: Path, save: bool = False) -> SPMPrediction:
        """Generates prediction for the given image and returns it as an SPMPrediction object.

        Args:
            image (Path): Path to the input image for prediction.
            save (bool): Whether to save the predictions.

        Returns:
            SPMPrediction: The prediction result containing polygons, class IDs, confidences, and bounding boxes.
        """
        prediction = SPMPrediction()

        border_prediction_idxs: list[int] = []
        inner_prediction_idxs: list[int] = []

        with rasterio.open(image) as src:
            # Get image height and width for effective overlap calculation
            img_width, img_height = src.width, src.height

            prediction_idx = 0
            # Stream tiles from the image and run inference on each tile
            for batch, coordinate in stream_tiles_by_batch(src, tile_size=self.config.tile_size, batch_size=self.config.batch_size, overlap=self.config.overlap):
                results = self.model(batch)

                for i, res in enumerate(results):
                    tile_overlap = effective_overlap(
                    tile_x=coordinate[i][1],
                    tile_y=coordinate[i][0],
                    tile_size=self.config.tile_size,
                    overlap=self.config.overlap,
                    img_width=img_width,
                    img_height=img_height,
                )
                    for j, (bbox, mask, cls, conf) in enumerate(zip(res.boxes.xyxy, res.masks.xy, res.boxes.cls, res.boxes.conf)):
                        if len(bbox) != 4 or len(mask) < 4:
                            logger.info(f"No boxes found in tile {i} at coordinate {coordinate[i]}")
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

                        if is_border_candidate(bbox=_bbox, tile_size=self.config.tile_size, overlap=tile_overlap):
                            border_prediction_idxs.append(prediction_idx)
                        else:
                            inner_prediction_idxs.append(prediction_idx)

                        prediction.add_annotation(
                            name=res.names[_cls],  # class label
                            class_id=_cls,
                            confidence=_conf,
                            bbox=_bbox,
                            segmentation=_mask
                        )
                        prediction_idx += 1

            # Merge border predictions using SPM
            smp = SpatialPolygonMerger()
            smp.index(prediction)
            merged_polygons = smp.merge(border_prediction_idxs)

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
        tile_size=3000,
        batch_size=4,
        overlap=300
    )
    model = YOLOModel(config)
    prediction = model.predict(Path("christchurch_487.tif"), save=True)