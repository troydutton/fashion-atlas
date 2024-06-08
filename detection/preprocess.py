import json
import os

from PIL import Image
from tqdm import tqdm


def preprocess_labels(meta_dir: str, image_dir: str, output_dir: str):
    # Verify the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a dictionary to store the label names
    labels = {}

    # Preprocess each json file in the source directory
    for input_file_name in tqdm(os.listdir(meta_dir), desc="Processing Labels", unit="file"):
        # Skip non-json files
        if not input_file_name.endswith(".json"):
            continue

        raw_file_name = input_file_name.replace(".json", "")

        input_file_path = os.path.join(meta_dir, input_file_name)

        with open(input_file_path) as input_file:
            data = json.load(input_file)

            output_file_path = os.path.join(output_dir, raw_file_name) + ".txt"

            with open(output_file_path, "w") as output_file:
                for obj in data:
                    # Skip non-dictionary objects
                    if type(data[obj]) is not dict:
                        continue

                    label = data[obj].get("category_id") - 1

                    if label not in labels:
                        labels[label] = data[obj].get("category_name")

                    x_min, y_min, x_max, y_max = data[obj].get("bounding_box")

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min

                    image_file_path = os.path.join(image_dir, raw_file_name) + ".jpg"
                    image = Image.open(image_file_path)

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
    os.makedirs(output_dir, exist_ok=True)

    # Move each image in the source directory
    for input_file_name in tqdm(os.listdir(image_dir), desc="Moving Images", unit="file"):
        # Skip non-jpg files
        if not input_file_name.endswith(".jpg"):
            continue

        input_file_path = os.path.join(image_dir, input_file_name)
        output_file_path = os.path.join(output_dir, input_file_name)

        # Move the image
        os.rename(input_file_path, output_file_path)

if __name__ == "__main__":
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