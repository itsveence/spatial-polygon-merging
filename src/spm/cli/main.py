import argparse

def cli_train(args):
    from spm.training.train_yolo import train
    train(
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch, 
        resume=args.resume, 
        model_path=args.model_path
        )
    
def cli_predict(args):
    from spm import YOLOModel, ModelConfig
    config = ModelConfig(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
        )
    model = YOLOModel(config)
    result = model.predict(image=args.image_path, merge=args.merge, merge_only_seam=args.merge_only_seam)
    return result

def main():
    parser = argparse.ArgumentParser(description="Spatial Polygon Merging CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Train command ----
    train_parser = subparsers.add_parser("train", help="Train the YOLO segmentation model")
    train_parser.add_argument("--data", type=str, default="data/whu_yolo_dataset-2/whu.yaml", help="Path to the dataset YAML file")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    train_parser.add_argument("--batch", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    train_parser.add_argument("--model-path", type=str, help="Path to a custom model checkpoint to start training from")

    # ---- Predict command ----
    predict_parser = subparsers.add_parser("predict", help="Run inference with the YOLO segmentation model")
    predict_parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
    predict_parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    predict_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    predict_parser.add_argument("--tile-size", type=int, default=1500, help="Size of each tile for inference")
    predict_parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold for detections")
    predict_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    predict_parser.add_argument("--overlap", type=float, default=0.25, help="Overlap between tiles")
    predict_parser.add_argument("--image-path", type=str, help="Path to the image for prediction")
    predict_parser.add_argument("--merge", action="store_true", help="Merge predictions")
    predict_parser.add_argument("--merge-only-seam", action="store_true", help="Merge only the seam area")

    args = parser.parse_args()

    # Dispatch to the appropriate function based on the command
    dispatch = {
        "train": cli_train,
        "predict": cli_predict
    }

    dispatch[args.command](args)

if __name__ == "__main__":
    main()