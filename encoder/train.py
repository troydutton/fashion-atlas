import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from wandb.wandb_run import Run

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]


class DressCodeDataset(Dataset):
    def __init__(
        self, root: str, pairs: str, transformations: transforms.Compose
    ) -> None:
        super().__init__()

        # Root directory of the dataset
        self.root = root

        # Model-garment pairs
        self.data = pd.read_csv(
            os.path.join(self.root, pairs),
            delimiter="\t",
            header=None,
            names=["model", "garment", "label"],
        )

        self.transformations = transformations

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

        # TODO: Hard negative mining
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

        anchor = self.transformations(anchor)
        positive = self.transformations(positive)
        negative = self.transformations(negative)

        return anchor, positive, negative


def evaluate(
    encoder: nn.Module,
    test_data: DataLoader,
    loss_fcn: nn.Module,
    epoch: int,
    device: str = "cpu",
) -> dict[str:float]:
    """
    Evaluate the encoder network on the test set.
    """

    encoder.eval()

    validation_loss = 0.0
    euclidean_distance_ap = 0.0
    euclidean_distance_an = 0.0
    similarity_ap = 0.0
    similarity_an = 0.0

    with torch.no_grad():
        for i, (anchor, positive, negative) in tqdm(
            enumerate(test_data),
            f"Evaluation {epoch + 1}",
            unit="batch",
            total=len(test_data),
        ):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_features = encoder(anchor)
            positive_features = encoder(positive)
            negative_features = encoder(negative)

            validation_loss += loss_fcn(
                anchor_features, positive_features, negative_features
            )

            euclidean_distance_ap += torch.norm(
                anchor_features - positive_features, dim=1
            ).sum()

            euclidean_distance_an += torch.norm(
                anchor_features - negative_features, dim=1
            ).sum()

            similarity_ap += torch.cosine_similarity(
                anchor_features, positive_features
            ).mean()
            similarity_an += torch.cosine_similarity(
                anchor_features, negative_features
            ).mean()

    return {
        "Val/Triplet Loss": validation_loss / len(test_data),
        "Val/Euclidean Distance Ratio": euclidean_distance_an / euclidean_distance_ap,
        "Val/Cosine Similarity Ratio": similarity_ap / similarity_an,
    }


def train_one_epoch(
    encoder: nn.Module,
    train_data: DataLoader,
    loss_fcn: nn.Module,
    optimizer: optim.Optimizer,
    logger: Run,
    epoch: int,
    device: str = "cpu",
):
    """
    Train the encoder network for one epoch.
    """

    encoder.train()

    for i, (anchor, positive, negative) in tqdm(
        enumerate(train_data),
        f"Training {epoch + 1}",
        unit="batch",
        total=len(train_data),
    ):
        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_features = encoder(anchor)
        positive_features = encoder(positive)
        negative_features = encoder(negative)

        batch_loss: torch.Tensor = loss_fcn(
            anchor_features, positive_features, negative_features
        )

        # Log loss to tensorboard
        logger.log(
            {
                "Train/Triplet Loss": batch_loss,
                "Train/Learning Rate": optimizer.param_groups[-1]["lr"],
            }
        )

        batch_loss.backward()
        optimizer.step()


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
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

    optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    logger = wandb.init(dir=log_dir, project="fashion-atlas", name=model_name)

    for epoch in range(epochs):

        train_one_epoch(
            encoder=encoder,
            train_data=train_data,
            loss_fcn=loss_fcn,
            optimizer=optimizer,
            logger=logger,
            epoch=epoch,
            device=device,
        )
        metrics = evaluate(
            encoder=encoder,
            test_data=test_data,
            loss_fcn=loss_fcn,
            epoch=epoch,
            device=device,
        )

        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1} {metrics_str}")

        logger.log(metrics)

        scheduler.step()

        torch.save(
            encoder.state_dict(),
            f"{os.path.join(output_dir, model_name)}/checkpoint-{epoch + 1}.pt",
        )

    logger.finish()


if __name__ == "__main__":
    # Set the seed
    torch.manual_seed(42)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    encoder = models.resnet50()

    encoder = encoder.to(device)

    # Load the data
    transformations = transforms.Compose(
        [transforms.Resize((256, 192)), transforms.ToTensor()]
    )

    train_data = DressCodeDataset(
        root=DRESSCODE_ROOT,
        pairs="train_pairs_cropped.txt",
        transformations=transformations,
    )

    test_data = DressCodeDataset(
        root=DRESSCODE_ROOT,
        pairs="test_pairs_paired_cropped.txt",
        transformations=transformations,
    )

    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=50, shuffle=False)

    loss_fcn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y),
        margin=1.5,
    )

    train(
        encoder=encoder,
        train_data=train_loader,
        test_data=test_loader,
        loss_fcn=loss_fcn,
        epochs=6,
        device=device,
        model_name="ResNet50 Cosine Similarity M=1.5",
    )
