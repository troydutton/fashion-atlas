import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm
from utils import build_encoder, get_transforms, set_random_seed

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

MODEL_PATH = "models/convnext-s.pt"

def precompute_dataset_features(
        encoder: nn.Module, 
        data: pd.DataFrame, 
        transformations: transforms.Compose, 
        device = torch.device
    ) -> Tuple[Dict[str, Tensor], Dict[str, np.ndarray]]:
    """
    Calculates the features for the entire dataset.

    Returns a dictionary of features and indices for each garment type.
    """
    features = {"upper_body": [], "lower_body": [], "dresses": []}
    feature_indices = {"upper_body": [], "lower_body": [], "dresses": []}

    encoder.eval()

    for i, (_, garment, label) in tqdm(enumerate(data.values), desc="Calculating Features", total=len(data), unit="image"):
        # Load in the garment image
        garment_image = Image.open(os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "cropped_images", garment)).convert("RGB")

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
    set_random_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read in the dataset
    train_data = pd.read_csv(os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"), delimiter="\t", header=None, names=["model", "garment", "label"])

    test_data = pd.read_csv(os.path.join(DRESSCODE_ROOT, "test_pairs_paired_cropped.txt"), delimiter="\t", header=None, names=["model", "garment", "label"])

    data = pd.concat([train_data, test_data])

    # Load in the encoder network
    encoder, _ = build_encoder(embedding_dim=1024, expander_dim=4096, device=device)

    if os.path.exists(MODEL_PATH):
        checkpoint_path = MODEL_PATH
    else:
        checkpoint_path = select_model()

    print(f"Loading {checkpoint_path}...")

    encoder.load_state_dict(torch.load(checkpoint_path))

    _, transformations = get_transforms()

    features, feature_indices = precompute_dataset_features(encoder=encoder, data=data, transformations=transformations, device=device)

    torch.save(features, os.path.join(DRESSCODE_ROOT, "features.pt"))
    torch.save(feature_indices, os.path.join(DRESSCODE_ROOT, "feature_indices.pt"))