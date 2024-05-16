import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

def precompute_dataset_features(
    encoder: nn.Module, data: pd.DataFrame, transformations: transforms.Compose
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    features = {"upper_body": [], "lower_body": [], "dresses": []}
    feature_indices = {"upper_body": [], "lower_body": [], "dresses": []}

    encoder.eval()

    for i, (_, garment, label) in tqdm(
        enumerate(data.values),
        desc="Calculating Features",
        total=len(data),
        unit="image",   
    ):
        # Load in the garment image
        garment_image = Image.open(
            os.path.join(
                DRESSCODE_ROOT, DIRECTORY_MAP[label], "cropped_images", garment
            )
        ).convert("RGB")

        # Apply the transformations
        garment_image = transformations(garment_image)

        # Calculate the features
        with torch.no_grad():
            garment_features = encoder(garment_image.unsqueeze(0).to(device)).cpu()

        # Get the features
        features[DIRECTORY_MAP[label]].append(garment_features)
        feature_indices[DIRECTORY_MAP[label]].append(i)

    features["upper_body"] = torch.cat(features["upper_body"])
    features["lower_body"] = torch.cat(features["lower_body"])
    features["dresses"] = torch.cat(features["dresses"])

    feature_indices["upper_body"] = np.array(feature_indices["upper_body"])
    feature_indices["lower_body"] = np.array(feature_indices["lower_body"])
    feature_indices["dresses"] = np.array(feature_indices["dresses"])

    return features, feature_indices

if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read in the dataset
    data = pd.read_csv(
        os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"),
        delimiter="\t",
        header=None,
        names=["model", "garment", "label"],
    )

    # Load in the encoder network
    encoder = models.resnet50()

    encoder.load_state_dict(
        torch.load("models\ResNet50 Cosine Similarity Margin=1\checkpoint-3.pt")
    )

    encoder = encoder.to(device)

    transformations = transforms.Compose(
        [transforms.Resize((256, 192)), transforms.ToTensor()]
    )

    features, feature_indices = precompute_dataset_features(encoder=encoder, data=data, transformations=transformations)

    torch.save(features, os.path.join(DRESSCODE_ROOT, "train_features.pt"))
    torch.save(feature_indices, os.path.join(DRESSCODE_ROOT, "train_feature_indices.pt"))