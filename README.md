# Spatial Polygon Merging (SPM)

A polygon-space post-processing stage for **tiled instance segmentation of high-resolution
geospatial imagery**. When an aerial raster is too large to segment in a single pass, it is cut
into overlapping tiles and each tile is segmented independently. Objects that straddle a tile
boundary come back as several disjoint fragments, and the overlap regions produce duplicate
detections. SPM reconciles both artefacts by working directly on the predicted **polygons**
rather than on rasterised masks:

- it indexes every predicted footprint in an STRtree spatial index;
- it associates fragments by a **proximity** criterion (a within-distance test) instead of by
  IoU, so it can join the near-zero-overlap fragments left by a seam;
- it chains transitive associations with a depth-bounded, class-consistent breadth-first search;
- it fuses each group by polygon union.

Because instances are kept as vertices rather than pixels, per-instance memory scales with
footprint complexity instead of footprint area, which is what lets the merge run on full urban
scenes that mask-space methods cannot fit in memory.

## Repository layout

```
src/spm/            # the installable `spm` package
  cli/              # train / predict command-line entry point
  config/           # ModelConfig and SPMConfig dataclasses
  core/             # SPMPrediction container and geometry helpers
  merging/          # SpatialPolygonMerger — the SPM algorithm
  models/           # YOLO segmentation wrapper with tiled inference
  io/               # raster (GeoTIFF) and vector (GeoPackage/GeoJSON) I/O
  preprocessing/    # dataset preparation and mask-to-polygon conversion
  training/         # YOLO fine-tuning
  visualization/    # prediction overlays
benchmark/          # experimental harness comparing SPM against baselines
utils/adapters.py   # cross-representation adapters (SPM / SMM / SAHI)
notebooks/          # exploration and benchmark notebooks + results
```

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). On Linux, PyTorch is pulled from
the CUDA 12.4 index; on macOS, from the CPU index (see `pyproject.toml`).

```bash
uv sync
```

This installs the `spm` package and all dependencies into a local virtual environment.

## Usage

### Command line

Run tiled inference on a GeoTIFF and merge the fragmented predictions with SPM:

```bash
uv run python -m spm.cli.main predict \
  --model-path runs/segment/yolo-seg-whu/weights/best.pt \
  --image-path path/to/scene.tif \
  --tile-size 1500 \
  --overlap 0.1 \
  --merge --merge-only-seam
```

`--merge` applies SPM; adding `--merge-only-seam` restricts the merge seeds to detections that
reach into a tile-overlap zone (faster, and the locality-preserving default). Omitting both
returns the raw, unmerged detections.

Fine-tune the YOLO segmentation backbone on a prepared dataset:

```bash
uv run python -m spm.cli.main train \
  --data data/whu_yolo_dataset/whu.yaml \
  --epochs 100 --imgsz 640 --batch 4
```

### As a library

```python
from spm import YOLOModel, ModelConfig, SpatialPolygonMerger, SPMConfig

# tiled inference
model = YOLOModel(ModelConfig(model_path="best.pt", tile_size=1500, overlap=0.1))
unmerged = model.predict("scene.tif", merge=False)

# merge in polygon space
merger = SpatialPolygonMerger(SPMConfig(tau_dist=1.0, tau_chain=5))
merger.index(unmerged)
merged = merger.merge(unmerged.seam_prediction_idxs)   # or merger.merge() to seed globally

merged.save_to_file(format="gpkg", output_dir="output/")
```

`SPMConfig` exposes the two merge parameters: `tau_dist`, the proximity threshold in pixels
within which two footprints are associated, and `tau_chain`, the maximum depth of transitive
chaining from a seed.

## Benchmarking

The `benchmark` package reproduces the controlled comparison from the dissertation. It runs
tiled inference once per test crop and applies every merging method — SPM, SMM, and the SAHI
post-processors (NMS, NMM, GreedyNMM, and LSNMS) — to that same set of detections, scoring each
on segmentation quality (mAP, precision, recall, F1) and computational cost (peak memory and
merge time), with each merge run in an isolated process so one method's memory cannot leak into
another's reading.

Generate a test set by cutting a GeoTIFF and its ground-truth shapefile into fixed-size crops:

```bash
uv run python -m benchmark generate scene.tif labels.shp test_set/ \
  --crop-sizes 5000 10000 20000 50000 100000
```

Evaluate every method over that test set:

```bash
uv run python -m benchmark evaluate test_set/ best.pt \
  --tile-size 1500 --overlap 0.1 --conf 0.3 --iou 0.5 \
  --output notebooks/results/benchmark_results.csv
```

A per-`(crop_size, method)` summary is written alongside the per-crop results.

## Dataset

Experiments use the [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
(Ji, Wei, & Lu, 2019): aerial imagery of Christchurch, New Zealand at 0.3 m ground sample
distance. Raster mask annotations are converted to the single-class YOLO segmentation label
format for training; the hold-out imagery is cut into fixed-size crops for the merge benchmark.
