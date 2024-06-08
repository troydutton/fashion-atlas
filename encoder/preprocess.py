import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

# Map labels to their corresponding segmentations (data/DressCode/readme.txt)
SEGMENT_MAP = [[4], [5, 6], [7]]

# Map classes to their corresponding labels (data/DeepFashion/DeepFashion2.yaml)
CLASS_MAP = [[0, 1, 2, 3, 4], [6, 7, 8], [9, 10, 11, 12]]


def get_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Get the bounding box which encompasses the given mask.

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

    # Load the model image and segmentation
    model_image = Image.open(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "images", model)
    ).convert("RGB")

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

    mask = np.isin(segmentation, SEGMENT_MAP[label])

    # Skip the image if the mask is empty
    if not mask.any():
        return False

    x_min, y_min, x_max, y_max = get_bounding_box(mask)

    model_image_cropped = model_image.crop((y_min, x_min, y_max, x_max))

    model_image_cropped.save(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "cropped_images", model)
    )

    return True


def crop_garment_image(garment: str, label: int, yolo: YOLO) -> bool:
    """
    Crop the garment image using YOLO's predicted bounding boxes. Saves the cropped image if successful.

    Returns True if the garment image was cropped successfully, False otherwise.
    """

    garment_image = Image.open(
        os.path.join(DRESSCODE_ROOT, DIRECTORY_MAP[label], "images", garment)
    ).convert("RGB")

    prediction_results = yolo.predict(garment_image, verbose=False)[0]

    # If there are no bounding boxes, skip the image
    if len(prediction_results) == 0:
        return False

    boxes = prediction_results.boxes

    classes = boxes.cls.cpu().numpy()

    correct_class_indices = np.where(np.isin(classes, CLASS_MAP[label]))[0]

    # If there are boxes of the correct class, we discard all other predictions
    if len(correct_class_indices) > 0:
        boxes = boxes[correct_class_indices]

    confidence = boxes.conf.cpu().numpy()

    # Choose the box with the highest confidence
    box = boxes[np.argmax(confidence)].xyxy.cpu().numpy().squeeze()

    garment_image_cropped = garment_image.crop(box)

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

    # Remove skipped images
    train_pairs = train_pairs[
        ~train_pairs["model"].isin([image[0] for image in skipped_images])
    ]

    test_pairs = test_pairs[
        ~test_pairs["model"].isin([image[0] for image in skipped_images])
    ]

    train_pairs.to_csv(
        os.path.join(DRESSCODE_ROOT, "train_pairs_cropped.txt"),
        sep="\t",
        header=False,
        index=False,
    )

    test_pairs.to_csv(
        os.path.join(DRESSCODE_ROOT, "test_pairs_paired_cropped.txt"),
        sep="\t",
        header=False,
        index=False,
    )

if __name__ == "__main__":
    preprocess_images()