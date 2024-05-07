# <img src="img/icon.png" alt="Image Description" width="50" /> Fashion Atlas


This project aims to address the challenge of cross-matching clothes worn by individuals in real life to a database of images. To achieve this we created an application that snaps a picture of a person, segments the different articles of clothing, and matches it to the closest unique feature encoding in our predefined database. We were able to generate good matches by fine-tuning an object detection model and training an encoder network using triplet margin loss. With our selection of models optimized for inference speed and accuracy, our application achieves low latency response times and precise recommendations.

Paper to Code: [Fashion Atlas](https://drive.google.com/file/d/18KIJtBHPAO6X9lUBsgYLL8pWhv92UXd6/view?usp=drive_link)

## Install

The installation instructions assume that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [node](https://nodejs.org/en/download) installed.

### Backend

1. Create a conda environment with the required dependencies:
```bash
conda env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate fashion-atlas
```

### Frontend

1. Install the required node packages:
```bash
cd app/frontend
npm install
```

2. Download Expo Go on phone App Store: 

<img src="img/expo-go.png" alt="Shirts" height="80" />

### Datasets

1. Download the DeepFashion2 Dataset: https://github.com/switchablenorms/DeepFashion2

2. Download the Dress Code Dataset: https://github.com/aimagelab/dress-code

## Run
1. Start NPX and save outputted host IP
```bash
cd app/frontend
npx expo start
```

2. Scan the printed QR code from the Expo App (Android) or the Camera App (Apple).

3. Open another terminal, and start server
```bash
cd app/backend
python server.py <host>
``` 