# benchmark

Build geo-referenced test crops from a large GeoTIFF + label shapefile, then
compare polygon-merging methods (SPM, SMM, SAHI, supervision) with pycocotools.

## 1. Generate a test set

```bash
python -m benchmark generate \
    "test_images/1.the whole aerial image.tif" labels.shp test_set \
    --crop-sizes 512 1024 2048 --max-crops 10
```

Each crop is written as:

```
test_set/crop_<size>_<i>/
    image.tif      # crop, carries its own geo-transform
    labels.gpkg    # ground-truth polygons clipped to the crop (raster CRS)
```

## 2. Benchmark merging methods

```bash
python -m benchmark evaluate test_set \
    --model weights/yolo26s-seg.pt --tile-size 1024 \
    --output benchmark_results.csv
```

For every crop the YOLO tiled inference runs once; the resulting raw detections
are fed to each merging method so the comparison isolates the merge step. Each
merged result is scored against the crop's ground truth.

## Metrics

`mAP`, `mAP50`, `mAP75` come from COCOeval (segmentation). Precision, recall and
F1 are computed at the chosen IoU (default 0.5) by greedy score-ordered matching
on the same RLE masks. Results are emitted per `(crop, method)` plus a
`(crop_size, method)` summary.

## Programmatic use

```python
from benchmark import generate_test_set, benchmark_test_set, summarize
from spm.config import ModelConfig

generate_test_set("big.tif", "labels.shp", "test_set", crop_sizes=[512, 1024])
rows = benchmark_test_set("test_set", ModelConfig(model_path="weights/yolo26s-seg.pt"))
for entry in summarize(rows):
    print(entry)
```

Methods live in `MERGING_METHODS`; pass a subset to `benchmark_test_set(methods=...)`.
Run from the repository root so the `spatial_mask_merging` submodule and `utils`
package resolve.
