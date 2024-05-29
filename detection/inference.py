from typing import List, Tuple

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

CLASS_TO_LABEL = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

CLASS_NAMES = [
    "Short Sleeve Top",
    "Long Sleeve Top",
    "Short Sleeve Outwear",
    "Long Sleeve Outwear",
    "Vest",
    "Sling",
    "Shorts",
    "Trousers",
    "Skirt",
    "Short Sleeve Dress",
    "Long Sleeve Dress",
    "Vest Dress",
    "Sling Dress",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
yolo = YOLO("models/yolov8m.pt")

def get_bounding_boxes(image: Image, min_confidence: float = 0.5) -> List[Tuple[List[int], int, str]]:
    """
    Returns a list of predictions containing the bounding box, label, and class name.
    """
    predictions: Results = yolo.predict(image)[0]

    if len(predictions) == 0:
        return None

    detections = predictions.boxes

    detections = detections[detections.conf > min_confidence]

    results = []

    for detection in detections:
        bounding_box = detection.xyxy[0].round().int().tolist()
        label = CLASS_TO_LABEL[detection.cls.int().item()]
        class_name = CLASS_NAMES[detection.cls.int().item()]

        results.append((bounding_box, label, class_name))

    return results