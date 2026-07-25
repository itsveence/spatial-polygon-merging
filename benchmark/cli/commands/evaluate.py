from pathlib import Path

import click

from benchmark import benchmark_test_set
from benchmark.runner import _write_csv, summarize
from spm.config import ModelConfig


@click.command(short_help="Evaluate a model on a benchmark test set")
@click.argument("model", type=str)
@click.argument("test_set_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the model on (e.g., 'cpu', 'cuda')",
)
@click.option(
    "--batch-size", type=int, default=4, help="Batch size for model inference"
)
@click.option(
    "--conf", type=float, default=0.3, help="Confidence threshold for detections"
)
@click.option(
    "--tile-size", type=int, default=640, help="Tile size for model inference"
)
@click.option(
    "--overlap", type=float, default=0, help="Overlap size for tiled inference"
)
@click.option("--iou", type=float, default=0.5, help="IoU threshold for evaluation")
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="Path to output CSV file for results",
)
@click.option(
    "--merge-count",
    type=int,
    default=5,
    help="Number of times to repeat each merge, for averaging time and memory",
)
@click.option(
    "--crop-limit",
    type=int,
    default=None,
    help="Limit the number of crops to evaluate (for testing purposes)",
)
def evaluate(
    # model parameters
    model: str,
    device: str,
    batch_size: int,
    conf: float,
    tile_size: int,
    overlap: float,
    # benchmark parameters
    test_set_dir: str,
    iou: float,
    output: str,
    merge_count: int,
    crop_limit: int | None = None,
) -> None:
    """Evaluate a model on a benchmark test set."""

    model_config = ModelConfig(
        model_path=model,
        device=device,
        batch_size=batch_size,
        confidence_threshold=conf,
        tile_size=tile_size,
        overlap=overlap,
    )

    rows = benchmark_test_set(
        test_set_dir=test_set_dir,
        model_config=model_config,
        iou_threshold=iou,
        output_csv=output,
        merge_count=merge_count,
        crop_limit=crop_limit,
    )

    summary = summarize(rows)
    if summary and output:
        _write_csv(
            summary,
            Path(output).with_name(Path(output).stem + "_summary.csv"),
        )
    for entry in summary:
        print(
            f"{entry['crop_size']:>6} {entry['method']:<18} "
            f"mAP={entry['mAP']:.3f} mAP50={entry['mAP50']:.3f} "
            f"P={entry['precision']:.3f} R={entry['recall']:.3f} F1={entry['f1']:.3f}"
        )
