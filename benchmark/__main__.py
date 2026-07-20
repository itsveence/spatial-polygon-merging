from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import benchmark_test_set, generate_test_set
from benchmark.runner import _write_csv, summarize
from spm.config.config import ModelConfig


def _cmd_generate(args: argparse.Namespace) -> None:
    generate_test_set(
        image_path=args.image,
        shapefile_path=args.shapefile,
        output_dir=args.output,
        crop_sizes=args.crop_sizes,
        stride_ratio=args.stride_ratio,
        max_crops_per_size=args.max_crops,
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    config = ModelConfig(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        confidence_threshold=args.conf,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    rows = benchmark_test_set(
        test_set_dir=args.test_set,
        model_config=config,
        iou_threshold=args.iou,
        output_csv=args.output,
    )
    summary = summarize(rows)
    if summary and args.output:
        _write_csv(
            summary,
            Path(args.output).with_name(Path(args.output).stem + "_summary.csv"),
        )
    for entry in summary:
        print(
            f"{entry['crop_size']:>6} {entry['method']:<18} "
            f"mAP={entry['mAP']:.3f} mAP50={entry['mAP50']:.3f} "
            f"P={entry['precision']:.3f} R={entry['recall']:.3f} F1={entry['f1']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Cut a GeoTIFF + shapefile into test crops")
    gen.add_argument("image", type=Path)
    gen.add_argument("shapefile", type=Path)
    gen.add_argument("output", type=Path)
    gen.add_argument("--crop-sizes", type=int, nargs="+", default=[5000, 10000, 20000])
    gen.add_argument("--stride-ratio", type=float, default=1.0)
    gen.add_argument("--max-crops", type=int, default=None)
    gen.set_defaults(func=_cmd_generate)

    ev = sub.add_parser("evaluate", help="Benchmark merging methods over a test set")
    ev.add_argument("test_set", type=Path)
    ev.add_argument("model", type=Path)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--batch-size", type=int, default=4)
    ev.add_argument("--conf", type=float, default=0.3)
    ev.add_argument("--tile-size", type=int, default=1500)
    ev.add_argument("--overlap", type=float, default=0.1)
    ev.add_argument("--iou", type=float, default=0.5)
    ev.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    ev.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
