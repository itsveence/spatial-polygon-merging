from pathlib import Path

import cv2
import numpy as np
import rasterio

from spm.core.prediction import SPMPrediction
from spm.utils.logger import logger


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
            cv2.fillPoly(overlay, [pts], color=FILL_COLOR,
                         lineType=cv2.LINE_AA)

        # Collect geometry to draw crisp elements after the blend.
        outlines = []
        annotations = []

        for poly, bbox, confidence, name in zip(
            prediction.polygons, prediction.bboxes, prediction.confidences, prediction.names
        ):
            if poly.is_empty:
                continue

            sub_polys = poly.geoms if poly.geom_type == "MultiPolygon" else [
                poly]
            for subpoly in sub_polys:
                if subpoly.is_empty:
                    continue
                pts = np.array(subpoly.exterior.coords, dtype=np.int32)
                _fill(pts)
                outlines.append(pts)

            annotations.append((bbox, f"{name}: {confidence:.2f}"))

        # --- Blend the fills back (only the segmentation is transparent) -------
        cv2.addWeighted(overlay, FILL_ALPHA, img_bgr,
                        1 - FILL_ALPHA, 0, dst=img_bgr)

        # --- Draw crisp, fully-opaque elements on top -------------------------
        # Polygon outlines give the segmentation a defined edge.
        for pts in outlines:
            cv2.polylines(img_bgr, [pts], isClosed=True, color=OUTLINE_COLOR,
                          thickness=OUTLINE_THICKNESS, lineType=cv2.LINE_AA)

        # Bounding boxes + labels with a filled background for readability.
        for bbox, label in annotations:
            x1, y1, x2, y2 = (int(bbox[0]), int(
                bbox[1]), int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR,
                          BOX_THICKNESS, lineType=cv2.LINE_AA)

            (tw, th), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS)
            # Keep the label inside the image if the box is near the top edge.
            label_y = y1 - 6 if y1 - th - 6 > 0 else y1 + th + 6
            bg_top = label_y - th - baseline
            cv2.rectangle(img_bgr, (x1, bg_top), (x1 + tw + 6, label_y + baseline),
                          BOX_COLOR, thickness=cv2.FILLED)
            cv2.putText(img_bgr, label, (x1 + 3, label_y), FONT, FONT_SCALE,
                        TEXT_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)

        cv2.imwrite(str(output_path), img_bgr)