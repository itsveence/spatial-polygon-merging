import click
from spm.training.train_yolo import train as train_yolo


@click.command(short_help="Train YOLO model")
@click.argument("data", type=click.Path(exists=True), required=True)
@click.argument("model", type=str)
@click.option("--epochs", type=int, default=100, help="Number of training epochs")
@click.option("--imgsz", type=int, default=640, help="Image size for training")
@click.option("--batch", type=int, default=4, help="Batch size for training")
@click.option("--resume", is_flag=True, help="Resume training from the last checkpoint")
def train(
    data,
    model,
    epochs,
    imgsz,
    batch,
    resume,
):
    click.echo("Training YOLO model with the following parameters:")
    click.echo(f"Data: {data}")
    click.echo(f"Model Path: {model}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Image Size: {imgsz}")
    click.echo(f"Batch Size: {batch}")
    click.echo(f"Resume: {resume}")

    # Call the train function from the training module
    train_yolo(
        data=data,
        model_path=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        resume=resume,
    )
