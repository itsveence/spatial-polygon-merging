# benchmark

Build geo-referenced test crops from a large GeoTIFF + label shapefile, then compare
polygon-merging methods (SPM, SMM, SAHI) on the same detections.

Run everything from the repository root so the `spatial_mask_merging` submodule resolves.

## 1. Generate a test set

```bash
uv run python -m benchmark generate \
    "test_images/1.the whole aerial image.tif" labels.shp test_set \
    --crop-sizes 512 --crop-sizes 1024 --crop-sizes 2048 --max-crops 10
```

`--crop-sizes` is repeatable (default `5000`, `10000`, `20000`); `--stride-ratio` sets the
window step as a fraction of the crop size (default `1.0`, non-overlapping) and `--max-crops`
caps how many crops are kept per size, densest first.

Each crop is written as:

```
test_set/crop_<size>_<i>/
    image.tif      # crop, carries its own geo-transform
    labels.gpkg    # ground-truth polygons clipped to the crop (raster CRS)
```

## 2. Benchmark merging methods

```bash
uv run python -m benchmark evaluate \
    weights/yolo26s-seg.pt test_set \
    --tile-size 1024 --overlap 0.1 --conf 0.3 --iou 0.5 \
    --merge-count 5 \
    --output benchmark_results.csv
```

Model weights are the first argument and the test-set directory the second. For every crop the
YOLO tiled inference runs once; the resulting raw detections are fed to each merging method so
the comparison isolates the merge step. Each merged result is scored against the crop's ground
truth and saved under `benchmark_output/<crop>/<method>/`.

Each merge runs in a spawned child process — pinned to a single core, with CUDA disabled and a
40 GB address-space cap — and its peak RSS is sampled from a monitor thread. That keeps one
method's residual memory out of the next method's reading and captures native allocations
(OpenCV, LSNMS, numpy) that a Python-level profiler cannot see. `--merge-count` (default 5)
repeats each merge so time and memory can be averaged; `--crop-limit` truncates the crop list
for a quick run.

## Methods

`MERGING_METHODS` maps a name to a configured `MergingMethod`:

| key | what it runs |
| --- | --- |
| `spm_seam_only` | SPM seeded only from detections touching a tile-overlap zone |
| `spm_global` | SPM seeded from every detection |
| `sahi_nms` | SAHI `NMSPostprocess` |
| `sahi_nmm` | SAHI `NMMPostprocess` |
| `sahi_greedy_nmm` | SAHI `GreedyNMMPostprocess` |
| `sahi_lsnms` | SAHI `LSNMSPostprocess` |
| `smm` | Spatial Mask Merging (submodule), at the Optuna-tuned parameters in `methods.py` |

`merge_supervision` (supervision's `with_nms` / `with_nmm`) is also implemented in `methods.py`
but is not registered in `MERGING_METHODS`; add a `MergingMethod` entry to include it.

## Metrics

`mAP`, `mAP50` and `mAP75` follow the COCO AP protocol over the 0.50–0.95 IoU sweep with 101
recall thresholds. Precision, recall and F1 come from greedy score-ordered matching at the
chosen IoU (default 0.5). All IoUs are computed on the polygon geometries — intersection area
over union area — with an STRtree over the ground truth pruning the pairs that need an exact
test, so nothing is rasterised during evaluation.

Results are emitted per `(crop, method)`, alongside a `(crop_size, method)` summary carrying
the mean, population standard deviation, min and max of every metric (`<key>`, `<key>_std`,
`<key>_min`, `<key>_max`).

## Programmatic use

```python
from benchmark import generate_test_set, benchmark_test_set, summarize, MERGING_METHODS
from spm.config import ModelConfig

generate_test_set("big.tif", "labels.shp", "test_set", crop_sizes=[512, 1024])

rows = benchmark_test_set(
    "test_set",
    ModelConfig(model_path="weights/yolo26s-seg.pt", tile_size=1024, overlap=0.1),
    methods=[MERGING_METHODS["spm_global"], MERGING_METHODS["sahi_nmm"]],
)
for entry in summarize(rows):
    print(entry)
```

`methods` takes `MergingMethod` objects, not names; omit it to run all of `MERGING_METHODS`.

## Tests

```bash
uv run pytest benchmark/tests
```
