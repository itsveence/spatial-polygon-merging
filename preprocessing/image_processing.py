from pathlib import Path
import cv2
import tifffile as tiff
from utils.logger import logger

def tiff_to_png(tiff_path: Path, png_path: Path) -> None:
    """Converts TIFF images to PNG format."""
    img = tiff.imread(tiff_path)
    
    # Convert GRAY to RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # Drop alpha channel if present

    cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":    # Example usage
    tiff_file = Path("data/test/image/0.tif")
    output_dir = Path("bin/data/test/image")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / (tiff_file.stem + ".png")
    logger.info(f"Converting {tiff_file} to png format at {png_dir}")
    tiff_to_png(tiff_file, png_dir)