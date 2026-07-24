import click

from benchmark import generate_test_set


@click.command(short_help="Cut a GeoTIFF + shapefile into test crops")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.argument("shapefile", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option(
    "--crop-sizes",
    type=int,
    multiple=True,
    default=[5000, 10000, 20000],
    help="List of crop sizes to generate",
)
@click.option(
    "--stride-ratio", type=float, default=1.0, help="Stride ratio for generating crops"
)
@click.option(
    "--max-crops",
    type=int,
    default=3,
    help="Maximum number of crops to generate per crop size",
)
def generate(
    image: str,
    shapefile: str,
    output_dir: str,
    crop_sizes: list[int],
    stride_ratio: float,
    max_crops: int,
) -> None:
    click.echo("Generating test crops with the following parameters:")
    click.echo(f"Image Path: {image}")
    click.echo(f"Shapefile Path: {shapefile}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Crop Sizes: {crop_sizes}")
    click.echo(f"Stride Ratio: {stride_ratio}")
    click.echo(f"Max Crops per Size: {max_crops}")

    generate_test_set(
        image_path=image,
        shapefile_path=shapefile,
        output_dir=output_dir,
        crop_sizes=crop_sizes,
        stride_ratio=stride_ratio,
        max_crops_per_size=max_crops,
    )
