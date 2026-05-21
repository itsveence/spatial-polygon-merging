from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
from utils.logger import logger

def tiff_to_png(tiff_path: Path, png_path: Path) -> None:
    """Converts TIFF images to PNG format."""
    try:
        img = tiff.imread(tiff_path)
        
        # Convert GRAY to RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Drop alpha channel if present

        cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.error(f"Failed to convert {tiff_path} to PNG: {e}")

def mask_to_txt(mask_path: Path, txt_path: Path, min_area: int = 20) -> None:
    """Converts mask images to text format."""
    try:
        mask = tiff.imread(mask_path)
        logger.debug(f"Loaded mask from {mask_path} with shape {mask.shape} and dtype {mask.dtype}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = ~mask # Invert mask: 0/False becomes 255/True, and vice versa

        binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} contours in {mask_path}: {contours}")

        h, w = binary.shape
        lines = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            points = approx.reshape(-1, 2)

            # normalize x,y coordinates
            normalized = []
            for x, y in points:
                normalized.extend([x / w, y / h])
            logger.debug(f"Contour with area {area} has {len(points)} points, normalized: {normalized}")
            line = f"{0} " + " ".join(f"{p:.6f}" for p in normalized)
            lines.append(line)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to convert {mask_path} to TXT: {e}")

if __name__ == "__main__":    # Example usage
    tiff_file = Path("data/test/image/0.tif")
    output_dir = Path("bin/data/test/image")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / (tiff_file.stem + ".png")
    logger.info(f"Converting {tiff_file} to png format at {png_dir}")
    tiff_to_png(tiff_file, png_dir)

    tiff_label = Path("data/test/label/1.tif")
    txt_dir = output_dir / (tiff_label.stem + ".txt")
    logger.info(f"Converting {tiff_label} to txt format at {txt_dir}")
    mask_to_txt(tiff_label, txt_dir)