import argparse
import json
import os
import sys

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python yolo.py <command>")
        sys.exit(1)

    if sys.argv[1] == "preprocess":
        # Preprocess the training data
        labels = preprocess_labels(
            meta_dir="data/DeepFashion2/train/annos",
            image_dir="data/DeepFashion2/train/image",
            output_dir="/home/tdutton/Programs/ECE379K/image-atlas/data/DeepFashion2Yolo/labels/train",
        )

        # Preprocess the validation data
        preprocess_labels(
            meta_dir="data/DeepFashion2/validation/annos",
            image_dir="data/DeepFashion2/validation/image",
            output_dir="/home/tdutton/Programs/ECE379K/image-atlas/data/DeepFashion2Yolo/labels/val",
        )

        # Print the labels
        print(f"Unique id-label pairs in the dataset: {len(labels)}\n")

        for id, name in labels.items():
            print(f"{id}: {name}")
    elif sys.argv[1] == "train":
        # Set the device
         # Load Model
        model = model = YOLO("yolov8m")

        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, help="Number of epochs (e.g. 10)" required=False, default=10)
        parser.add_argument("--batch_size", type=int, help="Batch size (e.g. 16)", required=False, default=16)
        parser.add_argument(
            "--device", type=str, help="Device (e.g. 0)", required=False, default="0"
        )
        args = parser.parse_args()

        # Train the model
        model.train(
            data="config/DeepFashion2.yaml",
            epochs=args.epochs,
            batch=args.batch_size,
            device=args.device,
            project="logs/YOLO",
        )
    else:
        print("Invalid command. Please use 'preprocess' or 'train'.")
        sys.exit(1)

    
    

   
