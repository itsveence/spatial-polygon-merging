from pathlib import Path
from preprocessing.image_processing import tiff_to_png, mask_to_txt
from utils.logger import logger

def prepare_dataset(input_dir: Path, output_dir: Path) -> None:

    for partition in ["train", "test", "val"]:
        image_dir = input_dir / partition / "image"
        label_dir = input_dir / partition / "label"

        output_image_dir = output_dir / "images" / partition
        output_label_dir = output_dir / "labels" / partition

        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_dir.glob("*.tif"):
            png_path = output_image_dir / (image_path.stem + ".png")
            logger.info(f"Converting {image_path} to png format at {png_path}")
            tiff_to_png(image_path, png_path)

            label_path = label_dir / image_path.name
            txt_path = output_label_dir / (image_path.stem + ".txt")
            logger.info(f"Converting {label_path} to txt format at {txt_path}")
            mask_to_txt(label_path, txt_path)


if __name__ == "__main__":
    input_dir = Path("data")
    output_dir = Path("bin/data")
    prepare_dataset(input_dir, output_dir)