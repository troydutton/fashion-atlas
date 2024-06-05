<p align="center">
    <img src="assets/icon.png" alt="Image Description" width="200" /> 
</p>

# Fashion Atlas

This project aims to address the challenge of cross-matching clothes worn by individuals in real life to a database of images. To achieve this we created an application that snaps a picture of a person, segments the different articles of clothing, and matches it to the closest unique feature encoding in our predefined database. We were able to generate good matches by fine-tuning an object detection model and training an encoder network using triplet margin loss. With our selection of models optimized for inference speed and accuracy, our application achieves low latency response times and precise recommendations. The write-up for our code can be found here: [Fashion Atlas](https://drive.google.com/file/d/18KIJtBHPAO6X9lUBsgYLL8pWhv92UXd6/view?usp=drive_link).

## Setup

The installation instructions assume that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [node](https://nodejs.org/en/download) installed.

1. Clone the repository: 
```bash
git clone https://github.com/troydutton/fashion-atlas.git
```

2. Create a conda environment with the required dependencies:
```bash
conda env create -f environment.yaml
```

3. Install the required frontend packages:
```bash
cd app/frontend
npm install
```

## Datasets

We selected Deepfashion2 and DressCode to train our object detection model and encoder.

## DeepFashion2

DeepFashion2 contains images of clothing items paired with corresponding bounding boxes and classes. We used these image-label pairs to fine-tune our object detection model. The dataset is available on GitHub: [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

## DressCode

DressCode provides a comprehensive collection of model-garment image pairs. These garment images show the clothing item against a uniform background, while the garment images show the same clothing item on a model. We use these images to generate anchor-positive pairs for training our encoder network. The DressCode dataset is available on GitHub: [DressCode](https://github.com/aimagelab/dress-code)

## Usage

1. Start NPX and save outputted host IP
```bash
cd app/frontend
npx expo start
```

2. Open another terminal, and start the server with host IP.
```bash
cd app/backend
python server.py <host>
``` 

3. Scan the QR code from the Expo App (Android) or the Camera App (Apple).

# Authors

Created during Spring 2024 for ECE379K (Computer Vision).
- Haakon Mongstad
- Jasper Tan
- Varun Arumugam
- Troy Dutton