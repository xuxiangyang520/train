from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
import os
import time
import argparse

def main(args):
    # 使用命令行参数而非硬编码值
    model_path = args.model_path
    data_path = args.data_path
    epochs = args.epochs
    lr0 = args.lr0
    batch_size = args.batch_size
    project = args.project
    name = args.name

    # 检查数据集配置文件
    if not os.path.exists(data_path):
        print(f"错误: 数据集配置文件不存在 - {data_path}")
        exit(1)

    # 加载模型（优先使用指定路径的模型，否则使用默认预训练模型）
    if os.path.exists(model_path):
        print(f"✅ 加载模型: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"⚠️ 指定的模型不存在，加载默认预训练模型 yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # 开始训练
    print(f"🚀 开始训练，总轮次: {epochs}，批量大小: {batch_size}")
    print(f"📁 结果将保存在: {os.path.join(project, name)}")
    start_time = time.time()

    results = model.train(
        data=data_path,
        epochs=epochs,
        lr0=lr0,
        batch=batch_size,
        resume=False,         # 从上次训练继续（若有模型路径）
        project=project,
        name=name,
    )

    # 训练完成统计
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f"✅ 训练完成！耗时: {elapsed_hours:.2f} 小时")
    print(f"📁 最终结果保存在: {os.path.join(project, name)}")

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--model_path', type=str, default="yolov8n.pt", help='模型路径')
    parser.add_argument('--data_path', type=str, default="D:/shujuji/yolov8/ultralytics-main/data.yml", help='数据集配置路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    parser.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--project', type=str, default="runs/detect", help='训练结果保存目录')
    parser.add_argument('--name', type=str, default="train34", help='训练任务名称')
    args = parser.parse_args()

    # 打印当前使用的配置
    print("\n===== 当前训练配置 =====")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=======================\n")

    # 执行主函数
    main(args)