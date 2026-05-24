import argparse

def cli_train(args):
    from training.yolo_seg_training import train
    train(
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch, 
        resume=args.resume, 
        model_path=args.model_path
        )

def main():
    parser = argparse.ArgumentParser(description="Spatial Polygon Merging CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Train command ----
    train_parser = subparsers.add_parser("train", help="Train the YOLO segmentation model")
    train_parser.add_argument("--data", type=str, default="bin/data/whu.yaml", help="Path to the dataset YAML file")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    train_parser.add_argument("--batch", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    train_parser.add_argument("--model-path", type=str, help="Path to a custom model checkpoint to start training from")

    args = parser.parse_args()

    # Dispatch to the appropriate function based on the command
    dispatch = {
        "train": cli_train,
    }

    dispatch[args.command](args)

if __name__ == "__main__":
    main()