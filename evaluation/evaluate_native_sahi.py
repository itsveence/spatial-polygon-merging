from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from utils.helpers import size_it, time_it
from utils.logger import logger

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/segment/yolo-seg-whu/weights/best.pt",  
    confidence_threshold=0.25,
    device="cuda:0",
)


@size_it
@time_it
def predict(image, model):
    return get_sliced_prediction(
        image, model,
        slice_height=1500,
        slice_width=1500,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        auto_slice_resolution=False,
        postprocess_type="NMM"
    )

image = "christchurch_487.tif"
detections = predict(image, detection_model)
logger.info(f"Total detections: {len(detections.object_prediction_list)}")
logger.info(f"Postprocessing time: {detections.durations_in_seconds}")
# detections.export_visuals(export_dir="demo_data/", hide_conf=True, hide_labels=True)