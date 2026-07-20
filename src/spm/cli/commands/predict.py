import click
from spm import YOLOModel, ModelConfig


@click.command(short_help="Predict using YOLO model")
@click.argument("model", type=click.Path(exists=True), required=True)
@click.argument("image", type=click.Path(exists=True), required=True)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the model on (e.g., 'cpu', 'cuda')",
)
@click.option("--batch-size", type=int, default=1, help="Batch size for prediction")
@click.option(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold for predictions",
)
@click.option(
    "--iou",
    type=float,
    default=0.45,
    help="IoU threshold for non-max suppression",
)
@click.option(
    "--tile-size",
    type=int,
    default=640,
    help="Size of the tiles to split the image into for prediction",
)
@click.option(
    "--overlap",
    type=float,
    default=0.2,
    help="Overlap between tiles when splitting the image",
)
@click.option("--merge", is_flag=True, help="Merge overlapping predictions")
@click.option("--merge-only-seam", is_flag=True, help="Merge only seam predictions")
@click.option(
    "--save-format",
    type=str,
    default="gpkg",
    help="Format to save the prediction output",
)
def predict(
    model,
    image,
    device,
    batch_size,
    conf,
    iou,
    tile_size,
    overlap,
    merge,
    merge_only_seam,
    save_format,
):
    click.echo("Running prediction with the following parameters:")
    click.echo(f"Model Path: {model}")
    click.echo(f"Image Path: {image}")
    click.echo(f"Device: {device}")
    click.echo(f"Batch Size: {batch_size}")
    click.echo(f"Confidence Threshold: {conf}")
    click.echo(f"IoU Threshold: {iou}")
    click.echo(f"Tile Size: {tile_size}")
    click.echo(f"Overlap: {overlap}")
    click.echo(f"Merge: {merge}")
    click.echo(f"Merge Only Seam: {merge_only_seam}")
    click.echo(f"Save Format: {save_format}")

    # Load the model
    config = ModelConfig(
        model_path=model,
        device=device,
        batch_size=batch_size,
        confidence_threshold=conf,
        iou_threshold=iou,
        tile_size=tile_size,
        overlap=overlap,
    )
    model = YOLOModel(config)

    # Run prediction
    predictions = model.predict(
        image=image, merge=merge, merge_only_seam=merge_only_seam
    )

    # Save prediction
    predictions.save_to_file(format=save_format)
