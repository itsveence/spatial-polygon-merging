import rasterio


def pixel_to_geo(transform, x, y):
        """Convert pixel coordinates to geographic coordinates using affine transformation."""
        geo_x, geo_y = transform * (x, y)
        return geo_x, geo_y

def get_crs_and_transform(image_path):
        if image_path is None:
            raise ValueError("image_path is not set for SPMPrediction.")
        with rasterio.open(image_path) as src:
            return src.crs, src.transform