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
src/spm/              # the installable `spm` package
  cli/                # `spm train` / `spm predict` command-line entry points
  config/             # ModelConfig, SPMConfig, and env-backed settings
  core/               # SPMPrediction container and geometry helpers
  merging/            # SpatialPolygonMerger — the SPM algorithm
  models/             # YOLO segmentation wrapper with tiled inference
  io/                 # raster (GeoTIFF) CRS/transform helpers
  preprocessing/      # dataset preparation and mask-to-polygon conversion
  training/           # YOLO fine-tuning
  visualization/      # prediction overlays
  utils/              # logging and memory/time profiling helpers
benchmark/            # experimental harness comparing SPM against baselines
  cli/                # `python -m benchmark generate | evaluate`
  crops.py            # test-set generation from a GeoTIFF + label shapefile
  methods.py          # the merging methods under test, each run in a child process
  eval.py             # polygon-IoU, COCO-style scoring
  adapters.py         # cross-representation adapters (SPM / SMM / SAHI)
  runner.py           # per-crop × per-method loop and summarisation
  tests/              # pytest unit tests
spatial_mask_merging/ # git submodule: the SMM baseline
notebooks/            # benchmark and SMM hyperparameter-tuning notebooks + results
```

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). On Linux, PyTorch is pulled from
the CUDA 12.4 index; on macOS, from the CPU index (see `pyproject.toml`).

```bash
git clone --recurse-submodules git@github.com:itsveence/spatial_polygon_merging.git
cd spatial_polygon_merging
uv sync
```

This installs the `spm` package (exposing the `spm` command-line entry point) and all
dependencies into a local virtual environment. If the repository was cloned without
`--recurse-submodules`, run `git submodule update --init` — the SMM baseline used by the
benchmark lives in the `spatial_mask_merging` submodule.

## Usage

### Command line

The package installs a Click-based `spm` command with two subcommands, `predict` and `train`.
Run it through uv (`uv run spm ...`) or from an activated environment (`spm ...`).

Run tiled inference on a GeoTIFF and merge the fragmented predictions with SPM. Both the model
weights and the image are positional arguments:

```bash
uv run spm predict \
  runs/segment/yolo-seg-whu/weights/best.pt \
  path/to/scene.tif \
  --tile-size 1500 \
  --overlap 0.1 \
  --conf 0.3 --iou 0.5 \
  --merge --merge-only-seam \
  --save-format gpkg
```

`--merge` applies SPM; adding `--merge-only-seam` restricts the merge seeds to detections that
reach into a tile-overlap zone (faster, and the locality-preserving default). Omitting both
returns the raw, unmerged detections. The result is written to `predictions/<format>/<image
stem>.<format>` — a GeoPackage or GeoJSON carrying the source raster's CRS.

Fine-tune a YOLO segmentation backbone on a prepared dataset. The dataset YAML and the base
model to fine-tune are positional arguments:

```bash
uv run spm train \
  data/whu_yolo_dataset/whu.yaml \
  yolo11n-seg.pt \
  --epochs 100 --imgsz 640 --batch 4
```

### As a library

```python
from spm import YOLOModel, ModelConfig, SpatialPolygonMerger, SPMConfig

# tiled inference
model = YOLOModel(ModelConfig(model_path="best.pt", tile_size=1500, overlap=0.1))
unmerged = model.predict("scene.tif", merge=False)

# merge in polygon space
merger = SpatialPolygonMerger(SPMConfig(tau_dist=1.0, rho_chain=5, score_agg="weighted_mean"))
merger.index(unmerged)
merged = merger.merge(unmerged.seam_prediction_idxs)   # or merger.merge() to seed globally

merged.save_to_file(format="gpkg", output_dir="output/")
```

`SPMConfig` exposes the merge parameters: `tau_dist`, the proximity threshold in pixels within
which two footprints are associated; `rho_chain`, the maximum depth of transitive chaining from
a seed; and `score_agg`, which decides the confidence carried by a fused instance.

`score_agg` is one of `"weighted_mean"` (the default, each fragment's score weighted by its
polygon area), `"mean"`, `"max"` or `"min"`. Because the fragments of one object are usually of
very unequal size, the area weighting keeps a small seam sliver with a poor score from dragging
down the confidence of the instance it belongs to.

## Benchmarking

The `benchmark` package reproduces the controlled comparison from the dissertation. It runs
tiled inference once per test crop and applies every merging method to that same set of raw
detections — SPM in both its seam-seeded and globally-seeded form, SMM, and the SAHI
post-processors (NMS, NMM, GreedyNMM, and LSNMS) — scoring each on segmentation quality (mAP,
mAP50, mAP75, precision, recall, F1) and computational cost (peak memory and merge time). Each
merge runs in a freshly spawned child process, pinned to one core and with CUDA disabled, so
that no method's memory can leak into another's reading and the peak-RSS figure covers native
allocations a Python-level profiler would miss.

Generate a test set by cutting a GeoTIFF and its ground-truth shapefile into fixed-size crops:

```bash
uv run python -m benchmark generate scene.tif labels.shp test_set/ \
  --crop-sizes 5000 --crop-sizes 10000 --crop-sizes 20000 \
  --max-crops 3
```

Evaluate every method over that test set — the model weights come first, then the test-set
directory:

```bash
uv run python -m benchmark evaluate best.pt test_set/ \
  --tile-size 1500 --overlap 0.1 --conf 0.3 --iou 0.5 \
  --merge-count 5 \
  --output notebooks/results/benchmark_results.csv
```

`--merge-count` repeats each merge that many times so the time and memory readings can be
averaged; `--crop-limit` truncates the crop list for a quick smoke run. One row per
`(crop, method)` goes to `--output`, and a per-`(crop_size, method)` summary — mean, population
standard deviation, min and max of every metric — is written alongside it as
`<output>_summary.csv`. The merged polygons themselves are saved under
`benchmark_output/<crop>/<method>/`.

Scoring is done directly on the polygons: mAP/mAP50/mAP75 follow the COCO AP protocol over the
0.50-0.95 IoU sweep, and precision/recall/F1 use greedy score-ordered matching at `--iou`. IoU
comes from the geometries themselves, with an STRtree over the ground truth pruning the
candidate pairs, so nothing is rasterised at evaluation time either.

`notebooks/benchmark.ipynb` turns the resulting CSVs into the figures under
`notebooks/results/`. `notebooks/smm_hyperparameter_tuning.ipynb` prepares predictions and
ground truth in the layout the SMM submodule expects and drives its `tools/optimize_smm.py`
Optuna search; its output (`notebooks/smm_optimization_files/opt_results/`) is where the tuned
SMM parameters in `benchmark/methods.py` come from, so that baseline is compared at its own
operating point rather than at its defaults.

## Development

Formatting and linting run through pre-commit (ruff plus mypy; the `spatial_mask_merging`
submodule is excluded from both):

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Tests are pytest:

```bash
uv run pytest
```

Runtime settings are read from a `.env` file at the repository root: `LOGGING_LEVEL`
(default `info`), `PROJECT_NAME` and `MODEL`.

## Dataset

Experiments use the [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
(Ji, Wei, & Lu, 2019): aerial imagery of Christchurch, New Zealand. Raster mask annotations are converted to the single-class YOLO segmentation label
format for training; the hold-out imagery is cut into fixed-size crops for the merge benchmark.

## License

Released under the [MIT License](LICENSE).
