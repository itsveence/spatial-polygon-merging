from pathlib import Path
from ultralytics.data.converter import convert_coco
from spm.preprocessing.image_processing import tiff_to_png, mask_to_txt
from spm.utils.logger import logger

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

def convert_coco_to_yolo(coco_json_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_coco(
        labels_dir=str(coco_json_path), 
        save_dir=str(output_dir), 
        use_segments=True, 
        use_keypoints=False, 
        cls91to80=False
        )

if __name__ == "__main__":
    # input_dir = Path("data")
    # output_dir = Path("bin/data")
    # prepare_dataset(input_dir, output_dir)

    coco_json_path = Path("data/whu_coco_dataset/annotation")
    output_dir = Path("data/whu_yolo_dataset")
    convert_coco_to_yolo(coco_json_path, output_dir)