import torch
from ultralytics import YOLO

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = YOLO("yolov8m")

    # Train the model
    model.train(
        data="config/DeepFashion2.yaml",
        epochs=50,
        batch=32,
        device=device,
        seed=42
    )
