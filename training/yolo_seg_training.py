import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from settings import PROJECT_NAME, MODEL

def train(data: str, epochs: int = 100, imgsz: int = 640, batch: int = 4, resume: bool = False, model_path: str = None) -> None:
    # Load a model
    if model_path:
        model = YOLO(model_path)  # load a custom model from a local path
    else:
        model = YOLO(MODEL)  # load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data=data, 
        epochs=epochs,
        imgsz=imgsz, 
        batch=batch,
        name=PROJECT_NAME, 
        exist_ok=True,
        resume=resume,
        )
    
