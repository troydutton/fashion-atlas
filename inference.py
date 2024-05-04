import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Root directory for the dataset
root = "data/DressCode/"

# Map labels to their corresponding directories
LABEL_TO_DIRECTORY = ["upper_body", "lower_body", "dresses"]

# Maps classes to their corresponding labels
CLASS_TO_LABEL = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

# Maps classes to their human-readable names
CLASS_TO_NAME = [
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear" "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]

# Set the seed
torch.manual_seed(42)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load in the encoder network
encoder = models.resnet50()

# Load the weights
encoder.load_state_dict(
    torch.load("models/ResNet50 Cosine Similarity Loss Margin 0.2/checkpoint-6.pt")
)

# Send the model to the device
encoder = encoder.to(device)

# Define the transformations for the network
transformations = transforms.Compose(
    [transforms.Resize((256, 192)), transforms.ToTensor()]
)

# Load the model
yolo = YOLO("models/yolov8m.pt")

# Read in the dataset
pairs = pd.read_csv(
    os.path.join(root, "train_pairs_cropped.txt"),
    delimiter="\t",
    header=None,
    names=["model", "garment", "label"],
)

features = torch.load("data/DressCode/train_features.pt")
feature_indices = torch.load("data/DressCode/train_feature_indices.pt")


def calculate_features(image: Image) -> np.ndarray:
    """
    Get the features for a given image.
    """
    # Set the model to evaluation mode
    encoder.eval()

    # Resize & convert to tensor
    image = transformations(image)

    # Add a batch dimension
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        return encoder(image).cpu()


def get_similar_images(image: Image, label: int, n: int = 5) -> list[Image.Image]:
    """
    Get the n most similar images to the given image.
    """

    # Caluclate the features for the image
    image_features = calculate_features(image)

    class_features = features[LABEL_TO_DIRECTORY[label]]

    # Calculate the cosine similarity for every image
    similarities = torch.cosine_similarity(image_features, class_features)

    # Find the n most similar images
    similar_image_indices = torch.argsort(similarities, descending=True)[:n].numpy()

    similar_image_indices = feature_indices[LABEL_TO_DIRECTORY[label]][
        similar_image_indices
    ]

    similar_images = []

    for idx in similar_image_indices:
        similar_images.append(
            Image.open(
                os.path.join(
                    root,
                    LABEL_TO_DIRECTORY[label],
                    "cropped_images",
                    pairs.iloc[idx]["garment"],
                )
            )
        )

    return similar_images


def display_similar_images(image: Image, similar_images: list[Image.Image]):
    """
    Display the similar images.
    """

    fig, axs = plt.subplots(1, len(similar_images) + 1, figsize=(20, 10))

    # Display the anchor image
    axs[0].imshow(image)
    axs[0].set_title("Anchor Image")

    for i, similar_image in enumerate(similar_images, 1):
        axs[i].imshow(similar_image)
        axs[i].set_title(f"Similar Image {i}")

    plt.show()


def infer(image: Image, min_confidence: float = 0.5) -> Image:
    # Get the predictions
    predictions = yolo.predict(image)[0]

    # Get the predicted detections
    detections = predictions.boxes

    # Threshold the predictions
    detections = detections[detections.conf > min_confidence]

    results = []

    for detection in detections:
        bounding_box = detection.xyxy.cpu().numpy().squeeze()

        image_cropped = image.crop(bounding_box)

        cls = detection.cls.int().item()
        label = CLASS_TO_LABEL[cls]

        similar_images = get_similar_images(image_cropped, label)

        results.append(
            {
                "bounding_box": bounding_box,
                "class": CLASS_TO_NAME[cls],
                "similar_images": similar_images,
            }
        )

        display_similar_images(image_cropped, similar_images)

    return results
