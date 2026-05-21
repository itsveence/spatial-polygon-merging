from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="bin/data/whu.yaml", 
    epochs=100, imgsz=640, 
    batch=4,
    name="yolo-seg-whu", 
    exist_ok=True
    )