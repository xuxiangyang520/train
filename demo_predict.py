
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict(source="./ultralytics/assets/bus,jpg")


