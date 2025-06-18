# 将代码保存为detect.py文件
from ultralytics import YOLO

model = YOLO("runs/detect/train20/weights/best.pt")
# 可以添加更多代码，例如预测或评估
results = model.predict(source="tests/datasets/sjj/images/test")
print(results)