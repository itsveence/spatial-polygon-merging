import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
import pytest
from pathlib import Path

from benchmark import generate_test_set


@pytest.fixture
def synthetic_spatial_data(tmp_path: Path):
    """Create a dummy 100x100 GeoTIFF and a dummy label Shapefile."""
    image_path = tmp_path / "dummy_image.tif"
    label_path = tmp_path / "dummy_labels.shp"

    # Create a dummy GeoTIFF
    transform = from_origin(0, 100, 1, 1)  # top-left corner at (0,100), pixel size 1x1
    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(image_path, "w", **profile) as dst:
        dst.write(np.random.randint(0, 255, (1, 100, 100), dtype=np.uint8))

    # Create a dummy label Shapefile with 2 polygons
    # Polygon 1 is in the top-left (hits the first crop)
    poly1 = Polygon([(10, 90), (20, 90), (20, 80), (10, 80)])
    # Polygon 2 is further right
    poly2 = Polygon([(60, 90), (70, 90), (70, 80), (60, 80)])

    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": [poly1, poly2]}, crs="EPSG:4326")
    gdf.to_file(label_path)

    return image_path, label_path


def test_generate_test_set_creates_valid_crops(synthetic_spatial_data, tmp_path: Path):
    image_path, label_path = synthetic_spatial_data
    output_dir = tmp_path / "crops_output"
    crop_sizes = [50]

    crops = generate_test_set(
        image_path=image_path,
        shapefile_path=label_path,
        output_dir=output_dir,
        crop_sizes=crop_sizes,
        stride_ratio=1.0,
        min_labels=1,
    )

    assert (
        len(crops) == 2
    ), "Expected 2 crops to be generated for the 100x100 image with crop size 50."

    first_crop = crops[0]
    assert first_crop.image_path.exists(), "Crop image file does not exist."
    assert first_crop.label_path.exists(), "Crop label file does not exist."
    assert first_crop.crop_size == 50, "Crop size does not match expected value."

    with rasterio.open(first_crop.image_path) as src:
        assert (
            src.width == 50 and src.height == 50
        ), "Crop image dimensions are incorrect."

    labels = gpd.read_file(first_crop.label_path)
    assert len(labels) >= 1, "Expected at least one label in the crop."
