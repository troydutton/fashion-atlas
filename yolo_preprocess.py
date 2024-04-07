import json
import os
from PIL import Image

import tqdm


def preprocess_yolo_labels():

    output_dir = "D:/Documents/Programs/ECE379K/datasets/deepfashion_yolo_format/labels"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for training data

    train_output_dir = os.path.join(output_dir, "train")
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)

    train_source_dir = "data/train/annos"
    train_image_source_dir = "D:/Documents/Programs/ECE379K/datasets/deepfashion_yolo_format/images/train"

    # loop through each json file in the source directory
    for file in tqdm.tqdm(os.listdir(train_source_dir)):

        if file.endswith(".json"):
            with open(os.path.join(train_source_dir, file)) as f:
                data = json.load(f)
                label_file = file.replace(".json", ".txt")
                with open(os.path.join(train_output_dir, label_file), "w") as out_file:
                    for obj in data:
                        if type(data[obj]) is not dict:
                            continue
                        label = data[obj].get("category_id") - 1
                        x1, y1, x2, y2 = data[obj].get("bounding_box")
                        # convert to x_center, y_center, width, height
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        # get total image height and width from image
                        # file_path = os.path.join(train_source_dir, file)
                        # image_file = file_path.replace(".json", ".jpg")
                        # image_file = image_file.replace("annos", "image")
                        # print(image_file)
                        # image_file = image_file.replace("\\", "/")
                        # print(image_file)

                        image = Image.open(os.path.join(train_image_source_dir, file.replace(".json", ".jpg")))
                        width, height = image.size

                        # normalize values
                        x_center /= width
                        y_center /= height
                        w /= width
                        h /= height

                        out_file.write(f"{label} {x_center} {y_center} {w} {h}\n")

    # for validation data

    val_output_dir = os.path.join(output_dir, "val")
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    val_source_dir = "data/validation/annos"
    val_image_source_dir = "D:/Documents/Programs/ECE379K/datasets/deepfashion_yolo_format/images/val"


    # loop through each json file in the source directory

    for file in tqdm.tqdm(os.listdir(val_source_dir)):

        if file.endswith(".json"):
            with open(os.path.join(val_source_dir, file)) as f:
                data = json.load(f)
                label_file = file.replace(".json", ".txt")
                with open(os.path.join(val_output_dir, label_file), "w") as out_file:
                    for obj in data:
                        if type(data[obj]) is not dict:
                            continue
                        label = data[obj].get("category_id") - 1
                        x1, y1, x2, y2 = data[obj].get("bounding_box")
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        # file_path = os.path.join(val_source_dir, file)
                        # image_file = file_path.replace(".json", ".jpg")
                        # image_file = image_file.replace("annos", "image")
                        # image_file = image_file.replace("\\", "/")

                        image = Image.open(os.path.join(val_image_source_dir, file.replace(".json", ".jpg")))
                        width, height = image.size

                        # normalize values
                        x_center /= width
                        y_center /= height
                        w /= width
                        h /= height

                        out_file.write(f"{label} {x_center} {y_center} {w} {h}\n")


def get_classes():
    # loop through training data to get all unique classes paired with their ids

    train_source_dir = "data/train/annos"

    classes = {}

    for file in os.listdir(train_source_dir):

        if file.endswith(".json"):
            with open(os.path.join(train_source_dir, file)) as f:
                data = json.load(f)
                for obj in data:
                    if type(data[obj]) is not dict:
                        continue
                    label = data[obj].get("category_id") - 1
                    if label not in classes:
                        classes[label] = data[obj].get("category_name")

    # sort classes by id
    classes = dict(sorted(classes.items()))

    with open("data/classes.txt", "w") as f:
        for key, value in classes.items():
            f.write(f"{key}: {value}\n")


preprocess_yolo_labels()