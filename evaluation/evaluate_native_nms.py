import supervision as sv
from PIL import Image
from ultralytics import YOLO

from utils.helpers import size_it, time_it

model = YOLO("runs/segment/yolo-seg-whu/weights/best.pt")

def callback(tile):
    results = model(tile)[0]
    return sv.Detections.from_ultralytics(results)

@size_it
@time_it
def slicer(image):
    _slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(1000, 1000),
        overlap_wh=(100, 100),
        iou_threshold=0.5,            # for merging duplicate detections across tiles
        overlap_filter=sv.OverlapFilter.NON_MAX_MERGE
    )
    return _slicer(image)

image = Image.open("christchurch_487.tif")
detections = slicer(image)

# detections.mask will contain the merged segmentation masks
# as a boolean array of shape (N, H, W)