import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

# Map labels to their corresponding segmentations (data/DressCode/readme.txt)
SEGMENT_MAP = [[4], [5, 6], [7]]

# Map labels to their corresponding classes (data/DeepFashion/DeepFashion2.yaml)
CLASS_MAP = [[0, 1, 2, 3, 4], [6, 7, 8], [9, 10, 11, 12]]

# Set the seed
torch.manual_seed(42)


class FashionDataset(Dataset):
    def __init__(self, root: str, pairs: str) -> None:
        super().__init__()

        self.transforms = transforms.Compose(
            [transforms.Resize((256, 192)), transforms.ToTensor()]
        )

        # Root directory of the dataset
        self.root = root

        # Load in the paired data
        self.data = pd.read_csv(
            pairs, delimiter="\t", header=None, names=["model", "garment", "label"]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        model, garment, label = self.data.iloc[index]

        # Load the anchor & positive images (random choice between model and garment)
        if random.choice([True, False]):
            anchor = Image.open(
                os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", model)
            ).convert("RGB")

            positive = Image.open(
                os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", garment)
            ).convert("RGB")
        else:
            anchor = Image.open(
                os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", garment)
            ).convert("RGB")

            positive = Image.open(
                os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", model)
            ).convert("RGB")

        # Randomly sample a negative (ensuring it is not the same as the anchor)
        while (negative_index := random.randrange(0, len(self.data))) == index:
            pass

        negative_model, negative_garment, negative_label = self.data.iloc[
            negative_index
        ]

        # Load the negative image (random choice between model and garment)
        if random.choice([True, False]):
            negative = Image.open(
                os.path.join(
                    self.root,
                    DIRECTORY_MAP[negative_label],
                    "cropped_images",
                    negative_garment,
                )
            ).convert("RGB")
        else:
            negative = Image.open(
                os.path.join(
                    self.root,
                    DIRECTORY_MAP[negative_label],
                    "cropped_images",
                    negative_model,
                )
            ).convert("RGB")

        # Resize & convert to tensors
        anchor = self.transforms(anchor)
        positive = self.transforms(positive)
        negative = self.transforms(negative)

        return anchor, positive, negative


def get_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Get the bounding box around the mask.

    Returns (x_min, y_min, x_max, y_max): The bounding box.
    """

    x_indices, y_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0, mask.shape[0], mask.shape[1]

    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))

    return x_min, y_min, x_max, y_max


def crop_model_image(model: str, label: int) -> bool:
    """
    Crop the model image using the corresponding segmentation. Saves the cropped image if successful.

    Returns True if the model image was cropped successfully, False otherwise.
    """

    # Load the model image
    model_image = Image.open(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "images", model)
    ).convert("RGB")

    # Load the segmentation
    segmentation = np.array(
        Image.open(
            os.path.join(
                DRESSCODE_ROOT,
                DIRECTORY_MAP[label],
                "label_maps",
                model.split("_")[0] + "_4.png",
            )
        )
    )

    # Get the mask for the label
    mask = np.isin(segmentation, SEGMENT_MAP[label])

    # Mask is empty, skip the image
    if not mask.any():
        return False

    # Get the bounding box for the mask
    x_min, y_min, x_max, y_max = get_bounding_box(mask)

    # Crop the image
    model_image_cropped = model_image.crop((y_min, x_min, y_max, x_max))

    # Save the cropped image
    model_image_cropped.save(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "cropped_images", model)
    )

    return True


def crop_garment_image(garment: str, label: int, yolo: YOLO) -> bool:
    """
    Crop the garment image using YOLO's predicted bounding boxes. Saves the cropped image if successful.

    Returns True if the garment image was cropped successfully, False otherwise.
    """

    # Load the garment image
    garment_image = Image.open(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "images", garment)
    ).convert("RGB")

    # Predict on the image
    prediction_results = yolo.predict(garment_image, verbose=False)[0]

    # If there are no bounding boxes, skip the image
    if len(prediction_results) == 0:
        return False

    # Get the predicted bounding boxes
    boxes = prediction_results.boxes

    # Get only the boxes that are of the correct class
    classes = boxes.cls.cpu().numpy()

    # Get the indices of the boxes that are of the correct class
    correct_class_indices = np.where(np.isin(classes, CLASS_MAP[label]))[0]

    # If there are boxes of the correct class, keep only those boxes
    if len(correct_class_indices) > 0:
        boxes = boxes[correct_class_indices]

    # Get the confidence score for each box
    confidence = boxes.conf.cpu().numpy()

    # Choose the box with the highest confidence
    box = boxes[np.argmax(confidence)].xyxy.cpu().numpy().squeeze()

    # Crop the image
    garment_image_cropped = garment_image.crop(box)

    # Save the cropped image
    garment_image_cropped.save(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "cropped_images", garment)
    )

    return True


def preprocess_images() -> None:
    """
    Preprocess the images for the DressCode dataset.

    Skips images that are not successfully cropped.
    """

    # Create output directories if they don't exist
    for directory in DIRECTORY_MAP:
        os.makedirs(
            os.path.join(DRESSCODE_ROOT, directory, "cropped_images"), exist_ok=True
        )

    # Read in the dataset
    train_pairs = pd.read_csv(
        os.path.join(DRESSCODE_ROOT, "train_pairs.txt"),
        delimiter="\t",
        header=None,
        names=["model", "garment", "label"],
    )

    test_pairs = pd.read_csv(
        os.path.join(DRESSCODE_ROOT, "test_pairs_paired.txt"),
        delimiter="\t",
        header=None,
        names=["model", "garment", "label"],
    )

    pairs = pd.concat([train_pairs, test_pairs])

    # Define an array to store skipped images
    skipped_images = []

    # Load in YOLO
    yolo = YOLO("models/yolov8m.pt")

    # Crop all the model & garment images
    for model, garment, label in tqdm(
        pairs.values, desc="Cropping Images", total=len(pairs), unit="image"
    ):
        # Crop the model image
        success = crop_model_image(model, label)

        if not success:
            skipped_images.append((model, garment, label))

        # Crop the garment image
        success = crop_garment_image(garment, label, yolo)

        if not success:
            skipped_images.append((model, garment, label))

    # Print the number of cropped images
    print(
        f"Successfully cropped {(1 - len(skipped_images) / len(pairs)) * 100:.2f}% of the dataset. ({len(skipped_images)} skipped)"
    )

    # Remove skipped images from the training set
    train_pairs = train_pairs[
        ~train_pairs["model"].isin([image[0] for image in skipped_images])
    ]

    # Save the training set
    train_pairs.to_csv(
        os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"),
        sep="\t",
        header=False,
        index=False,
    )

    # Remove skipped images from the test set
    test_pairs = test_pairs[
        ~test_pairs["model"].isin([image[0] for image in skipped_images])
    ]

    # Save the test set
    test_pairs.to_csv(
        os.path.join(DRESSCODE_ROOT, "test_pairs_paired_cropped.txt"),
        sep="\t",
        header=False,
        index=False,
    )


def train(
    encoder: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    loss_fcn: nn.Module,
    epochs: int = 10,
    device: str = "cpu",
    log_dir: str = "./logs",
    output_dir: str = "./models",
    model_name: str = "ResNet50",
):
    # Create the log & output directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Wandb logger
    logger = wandb.init(dir=log_dir, project="fashion-atlas", name=model_name)

    for epoch in range(epochs):

        # Set model to training mode
        encoder.train()

        for i, (anchor, positive, negative) in tqdm(
            enumerate(train_data),
            f"Epoch {epoch} Training",
            unit="batch",
            total=len(train_data),
        ):
            optimizer.zero_grad()

            # Send images to the device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass
            anchor_features = encoder(anchor)
            positive_features = encoder(positive)
            negative_features = encoder(negative)

            # Compute the loss
            loss = loss_fcn(anchor_features, positive_features, negative_features)

            # Log loss to tensorboard
            logger.log(
                {
                    "Train/Triplet Loss": loss,
                    "Train/Learning Rate": optimizer.param_groups[-1]["lr"],
                }
            )

            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate the model on the testing data
        encoder.eval()

        validation_loss = 0.0
        euclidean_distance_ap = 0.0
        euclidean_distance_an = 0.0
        similarity_ap = 0.0
        similarity_an = 0.0

        with torch.no_grad():
            for i, (anchor, positive, negative) in tqdm(
                enumerate(test_data),
                f"Epoch {epoch} Evaluation",
                unit="batch",
                total=len(test_data),
            ):
                # Send images to the device
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                # Forward pass
                anchor_features = encoder(anchor)
                positive_features = encoder(positive)
                negative_features = encoder(negative)

                # Compute the loss
                validation_loss += loss_fcn(
                    anchor_features, positive_features, negative_features
                )

                # Compute the Euclidean distance for the positive and negative pairs
                euclidean_distance_ap += torch.norm(
                    anchor_features - positive_features, dim=1
                ).sum()

                euclidean_distance_an += torch.norm(
                    anchor_features - negative_features, dim=1
                ).sum()

                # Compute the Cosine similarity for the positive and negative pairs
                similarity_ap += torch.cosine_similarity(
                    anchor_features, positive_features
                ).mean()
                similarity_an += torch.cosine_similarity(
                    anchor_features, negative_features
                ).mean()

        # Log validation metrics
        logger.log(
            {
                "Val/Triplet Loss": validation_loss / len(test_data),
                "Val/Euclidean Distance Ratio (AN % AP)": euclidean_distance_an
                / euclidean_distance_ap,
                "Val/Cosine Similarity Ratio (AP % AN)": similarity_ap / similarity_an,
            }
        )

        print(f"Epoch {epoch} Validation Loss: {validation_loss / len(test_data)}")

        # Save the model
        torch.save(
            encoder.state_dict(),
            f"{os.path.join(output_dir, model_name)}/checkpoint-{epoch + 1}.pt",
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python encoder.py <command>")
        sys.exit(1)

    if sys.argv[1] == "preprocess":
        preprocess_images()
    elif sys.argv[1] == "train":
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        encoder = models.resnet50()

        encoder = encoder.to(device)

        # Load the dataset
        train_data = FashionDataset(
            DRESSCODE_ROOT, "data/DressCode/train_pairs_cropped.txt"
        )

        test_data = FashionDataset(
            DRESSCODE_ROOT, "data/DressCode/test_pairs_paired_cropped.txt"
        )

        # Define the training dataloader
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Define the validation dataloader
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        train(
            encoder,
            train_loader,
            test_loader,
            nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y),
                margin=1,
            ),
            epochs=3,
            device=device,
            model_name="Test",
        )
    else:
        print("Invalid command. Please use 'preprocess' or 'train'.")
        sys.exit(1)
