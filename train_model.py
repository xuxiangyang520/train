from ultralytics import YOLO
import os
import time

# 配置参数
model_path = "runs/detect/train20/weights/last.pt"  # 或best.pt，首次训练可改为yolov8n.pt
data_path = "data.yml"
epochs = 300          # 总训练轮次
lr0 = 0.001           # 初始学习率
batch_size = 16       # 批量大小（根据显存调整）
project = "runs/detect"  # 训练结果保存目录
name = "train20_continued"  # 训练任务名称

# 检查数据集配置文件
if not os.path.exists(data_path):
    print(f"错误: 数据集配置文件不存在 - {data_path}")
    exit(1)

# 加载模型
if os.path.exists(model_path):
    print(f"✅ 加载继续训练的模型: {model_path}")
    model = YOLO(model_path)
else:
    print(f"⚠️ 未找到继续训练的模型，加载默认模型 yolov8n.pt")
    model = YOLO("yolov8n.pt")  # 首次训练使用预训练模型

# 开始训练
print(f"🚀 开始训练，总轮次: {epochs}，批量大小: {batch_size}")
start_time = time.time()

results = model.train(
    data=data_path,
    epochs=epochs,
    lr0=lr0,
    batch=batch_size,
    resume=True,         # 从上次训练继续（若有模型路径）
    project=project,
    name=name,
    # 可选参数（根据需求添加）
    # patience=10,       # 早停策略：连续10轮无提升则停止
    # val=True,          # 是否在训练中验证
    # save_period=10,    # 每10轮保存一次模型
)

# 训练完成统计
end_time = time.time()
elapsed_hours = (end_time - start_time) / 3600
print(f"✅ 训练完成！耗时: {elapsed_hours:.2f} 小时")
print(f"📁 结果保存在: {os.path.join(project, name)}")
print(f"📊 最佳模型: {os.path.join(project, name, 'weights/best.pt')}")