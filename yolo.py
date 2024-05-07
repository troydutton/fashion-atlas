import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Set the seed
torch.manual_seed(42)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_labels(meta_dir: str, image_dir: str, output_dir: str):
    # Verify the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a dictionary to store the label names
    labels = {}

    # Preprocess each json file in the source directory
    for input_file_name in tqdm(
        os.listdir(meta_dir), desc="Processing Labels", unit="file"
    ):
        # Skip non-json files
        if not input_file_name.endswith(".json"):
            continue

        raw_file_name = input_file_name.replace(".json", "")

        input_file_path = os.path.join(meta_dir, input_file_name)

        # Open the input file
        with open(input_file_path) as input_file:
            data = json.load(input_file)

            output_file_path = os.path.join(output_dir, raw_file_name) + ".txt"

            # Create the output file
            with open(output_file_path, "w") as output_file:
                for obj in data:
                    # Skip non-dictionary objects
                    if type(data[obj]) is not dict:
                        continue

                    # Get the bounding box label
                    label = data[obj].get("category_id") - 1

                    # Store the label name
                    if label not in labels:
                        labels[label] = data[obj].get("category_name")

                    # Get bounding box coordinates
                    x_min, y_min, x_max, y_max = data[obj].get("bounding_box")

                    # Convert to x_center, y_center, width, height.
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min

                    # Read the image
                    image_file_path = os.path.join(image_dir, raw_file_name) + ".jpg"

                    image = Image.open(image_file_path)

                    # Get the image dimensions
                    width, height = image.size

                    # Normalize values relative to image size
                    x_center /= width
                    y_center /= height
                    w /= width
                    h /= height

                    # Write the data to the output file
                    output_file.write(f"{label} {x_center} {y_center} {w} {h}\n")

    # Sort the labels by id
    labels = dict(sorted(labels.items()))

    return labels


def move_images(image_dir: str, output_dir: str):
    # Verify the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Move each image in the source directory
    for input_file_name in tqdm(
        os.listdir(image_dir), desc="Moving Images", unit="file"
    ):
        # Skip non-jpg files
        if not input_file_name.endswith(".jpg"):
            continue

        input_file_path = os.path.join(image_dir, input_file_name)
        output_file_path = os.path.join(output_dir, input_file_name)

        # Move the image
        os.rename(input_file_path, output_file_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python yolo.py <command>")
        sys.exit(1)

    if sys.argv[1] == "preprocess":
        # Preprocess the training data
        labels = preprocess_labels(
            meta_dir="data/DeepFashion2/train/annos",
            image_dir="data/DeepFashion2/train/image",
            output_dir="data/DeepFashion2Yolo/labels/train",
        )

        # Preprocess the validation data
        preprocess_labels(
            meta_dir="data/DeepFashion2/validation/annos",
            image_dir="data/DeepFashion2/validation/image",
            output_dir="data/DeepFashion2Yolo/labels/val",
        )

        # Move the training images
        move_images(
            image_dir="data/DeepFashion2/train/image",
            output_dir="data/DeepFashion2Yolo/images/train",
        )

        # Move the validation images
        move_images(
            image_dir="data/DeepFashion2/validation/image",
            output_dir="data/DeepFashion2Yolo/images/val",
        )

        # Print the labels
        print(f"Unique id-label pairs in the dataset: {len(labels)}\n")

        for id, name in labels.items():
            print(f"{id}: {name}")
    elif sys.argv[1] == "train":
        # Load Model
        model = model = YOLO("yolov8m")

        # Train the model
        model.train(
            data="config/DeepFashion2.yaml",
            epochs=10,
            batch=16,
            device=device,
        )
    else:
        print("Invalid command. Please use 'preprocess' or 'train'.")
        sys.exit(1)
