import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import yaml
from torchvision import models

IMAGE_SIZE = (256, 192)
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

def parse_config(path: str) -> Dict:
    """
    Reads a yaml file and returns a dictionary with the configuration parameters.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def build_encoder(embedding_dim: int, expander_dim: int, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Build the encoder and expander networks.
    """
    encoder = models.convnext_base(weights="DEFAULT").to(device)

    encoder.classifier[2] = nn.Linear(encoder.classifier[2].in_features, embedding_dim).to(device)

    expander = nn.Sequential(
        nn.Linear(embedding_dim, expander_dim),
        nn.BatchNorm1d(expander_dim),
        nn.ReLU(True),
        nn.Linear(expander_dim, expander_dim),
        nn.BatchNorm1d(expander_dim),
        nn.ReLU(True),
        nn.Linear(expander_dim, expander_dim),
    ).to(device)

    return encoder, expander

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the train and test transformations.
    """
    train_transformations = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(IMAGE_SIZE), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomPerspective(interpolation=transforms.InterpolationMode.NEAREST, fill=(0.9536, 0.9470, 0.9417))]),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))]),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )

    test_transformations = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(IMAGE_SIZE), 
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )

    return train_transformations, test_transformations

def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)