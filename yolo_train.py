from ultralytics import YOLO
import torch

model = YOLO('yolov5s.pt')  # Load YOLOv5s

print(torch.cuda.is_available())

model.train(data='data/yamls/deepfashion2.yaml', epochs=10, batch=32, device='0', workers=0)