import os

import pandas as pd
import torch
from PIL import Image

from .utils import build_encoder, get_transforms, parse_config

DRESSCODE_ROOT = "data/DressCode/"

LABEL_TO_GARMENT_TYPE = ["upper_body", "lower_body", "dresses"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_config("config/DressCode.yaml")

# Load the model
encoder, expander = build_encoder(**args["model"], device=device)

encoder.load_state_dict(torch.load("models/ConvNeXt-B 2024-05-27-21-18-57/checkpoint-2.pt", map_location=device))

encoder.eval()

_, transformations = get_transforms()

# Read in the dataset
pairs = pd.read_csv(os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"), delimiter="\t", header=None, names=["model", "garment", "label"])

# Load the features
features = torch.load(os.path.join(DRESSCODE_ROOT, "train_features.pt"))
feature_indices = torch.load(os.path.join(DRESSCODE_ROOT, "train_feature_indices.pt"))

def get_similar_images(image: Image, label: int, n: int = 4) -> list[Image.Image]:
    """
    Returns the n images that are most similar to the input image.
    """
    class_features = features[LABEL_TO_GARMENT_TYPE[label]]
    class_feature_indices = feature_indices[LABEL_TO_GARMENT_TYPE[label]]

    with torch.no_grad():
        # Caluclate the features for the image
        image: torch.Tensor = transformations(image).to(device)

        image_features = encoder(image.unsqueeze(0)).cpu()

        # Calculate the cosine similarity for every image
        similarities = torch.cosine_similarity(image_features, class_features)

        # Find the images with the highest similarity
        similar_image_indices = torch.argsort(similarities, descending=True)[:n].numpy()

        similar_image_indices = class_feature_indices[similar_image_indices]

        similar_images = [Image.open(os.path.join(DRESSCODE_ROOT, LABEL_TO_GARMENT_TYPE[label], "cropped_images", pairs.iloc[idx]["garment"])) for idx in similar_image_indices]

    return similar_images