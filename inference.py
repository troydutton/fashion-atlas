import os
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from encoder.utils import build_encoder, get_transforms, parse_config

DRESSCODE_ROOT = "data/DressCode/"

LABEL_TO_GARMENT_TYPE = ["upper_body", "lower_body", "dresses"]

CLASS_TO_LABEL = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_config("config/DressCode.yaml")

# Load the models
encoder, expander = build_encoder(**args["model"], device=device)

encoder.load_state_dict(torch.load("models/ConvNeXt-T Semi-Hard 2024-05-31-01-55-38/checkpoint-20.pt", map_location=device))

encoder.eval()

yolo = YOLO("models/yolov8m.pt")

_, transformations = get_transforms()

# Read in the dataset
train_data = pd.read_csv(os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"), delimiter="\t", header=None, names=["model", "garment", "label"])

test_data = pd.read_csv(os.path.join(DRESSCODE_ROOT, "test_pairs_paired_cropped.txt"), delimiter="\t", header=None, names=["model", "garment", "label"])

data = pd.concat([train_data, test_data])

# Load the features
features = torch.load(os.path.join(DRESSCODE_ROOT, "features.pt"))
feature_indices = torch.load(os.path.join(DRESSCODE_ROOT, "feature_indices.pt"))

def get_similar_garments(image: Image.Image, min_confidence: float = 0.5) -> List[Dict]:
    """
    Detects the garments in the image and returns the most similar garments from the dataset.

    For each garment it returns a dictionary which contains the following keys:

    bounding_box: The bounding box coordinates of the detected garment in image-normalized xyxy format.
    class_name: The human-readable class name of the detected garment.
    similar_garments: A list of images of similar garments against a plain background.
    similar_models: A list of images of models wearing the similar garments.
    """
    predictions: Results = yolo.predict(image)[0]

    if len(predictions) == 0:
        return None
    
    # Visualize predictions
    predictions.show()

    detections = predictions.boxes

    detections = detections[detections.conf > min_confidence]

    if len(detections) == 0:
        return None
    
    height, width = detections.orig_shape

    print(detections)

    results = []

    for detection in detections:
        bounding_box = detection.xyxy[0].round().int().tolist()
        
        label = CLASS_TO_LABEL[detection.cls.int().item()]
        class_name = CLASS_NAMES[detection.cls.int().item()]

        similar_garments, similar_models = get_similar_images(image.crop(bounding_box), label)

        print(similar_garments[0].size)

        results.append({
            "bounding_box": detection.xyxyn[0].tolist(),
            "class_name": class_name,
            "similar_garments": similar_garments,
            "similar_models": similar_models
        })

    return results

def get_similar_images(image: Image.Image, label: int, n: int = 4) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    Returns the top n most similar garment & model images for the given image.
    """
    class_features = features[LABEL_TO_GARMENT_TYPE[label]]
    class_feature_indices = feature_indices[LABEL_TO_GARMENT_TYPE[label]]

    with torch.no_grad():
        image: torch.Tensor = transformations(image).to(device)

        image_features = encoder(image.unsqueeze(0)).cpu()

        similarities = torch.cosine_similarity(image_features, class_features)

        similar_image_indices = torch.argsort(similarities, descending=True)[:n].numpy()

        similar_image_indices = class_feature_indices[similar_image_indices]

        similar_garments = [Image.open(os.path.join(DRESSCODE_ROOT, LABEL_TO_GARMENT_TYPE[label], "images", data.iloc[i]["garment"])) for i in similar_image_indices]

        similar_models = [Image.open(os.path.join(DRESSCODE_ROOT, LABEL_TO_GARMENT_TYPE[label], "images", data.iloc[i]["model"])) for i in similar_image_indices]

    return similar_garments, similar_models