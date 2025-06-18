from ultralytics import YOLO

# 加载训练好的模型（替换为你的模型路径）
model = YOLO("runs/detect/train34/weights/best.pt")

# 对单张图像进行检测
results = model.predict(
    source="tests/datasets/ceshi",
    save=True,
    save_txt=True,  # 保存检测框坐标到txt文件
    save_conf=True,  # 保存置信度分数
    conf=0.3,        # 降低阈值可检测更多目标
    iou=0.5,         # 交并比阈值，用于非极大值抑制
    line_thickness=2,  # 边界框线条粗细
    show_labels=True,  # 显示标签
    show_conf=True,    # 显示置信度
)
# 对视频或摄像头进行检测
# model.predict(source="path/to/video.mp4", save=True)  # 视频文件
# model.predict(source=0, save=True)  # 0表示默认摄像头