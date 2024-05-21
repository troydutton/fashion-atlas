import os
import random
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import wandb
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from wandb.wandb_run import Run

# Root directory for the dataset
DRESSCODE_ROOT = "data/DressCode/"

# Map labels to their corresponding directories
DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]


class DressCodeDataset(Dataset):
    def __init__(self, root: str, pairs: str, transformations: transforms.Compose) -> None:
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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        model, garment, label = self.data.iloc[index]

        # Load the anchor & positive images (random choice between model and garment)
        if random.choice([True, False]):
            anchor = Image.open(os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", model)).convert("RGB")

            positive = Image.open(os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", garment)).convert("RGB")
        else:
            anchor = Image.open(os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", garment)).convert("RGB")

            positive = Image.open(os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", model)).convert("RGB")

        # TODO: Hard negative mining
        # Randomly sample a negative (ensuring it is not the same as the anchor)
        while (negative_index := random.randrange(0, len(self.data))) == index:
            pass

        negative_model, negative_garment, negative_label = self.data.iloc[negative_index]

        # Load the negative image (random choice between model and garment)
        if random.choice([True, False]):
            negative = Image.open(os.path.join(self.root, DIRECTORY_MAP[negative_label], "cropped_images", negative_garment,)).convert("RGB")
        else:
            negative = Image.open(os.path.join(self.root, DIRECTORY_MAP[negative_label], "cropped_images", negative_model)).convert("RGB")

        anchor = self.transformations(anchor)
        positive = self.transformations(positive)
        negative = self.transformations(negative)

        return anchor, positive, negative
class EncoderLoss(nn.Module):
    def __init__(self, expander: nn.Module, triplet_weight: float = 1.0, vicreg_weight: float = 1.0, margin: float = 1.0) -> None:
        super().__init__()

        self.expander = expander
        self.triplet_weight = triplet_weight
        self.vicreg_weight = vicreg_weight

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y),
            margin=margin,
        )
        
        self.vicreg_loss = VICRegLoss()

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Dict[str, Tensor]:
        """
        Calculate the contrastive loss between the anchor, positive and negative samples. 
        Incorporates VICReg loss to prevent information collapse and encourage diversity in the embeddings.

        Returns a dictionary containing the triplet loss and VICReg loss.
        """

        # Calculate the triplet loss
        triplet_loss = self.triplet_loss(anchor, positive, negative)

        # Calculate the VICReg loss
        vicreg_loss = self.vicreg_loss(self.expander(anchor), self.expander(positive))

        # Calculate the overall loss
        overall_loss = self.triplet_weight * triplet_loss + self.vicreg_weight * vicreg_loss

        return {"Overall Loss": overall_loss, "Triplet Loss": triplet_loss, "VICReg Loss": vicreg_loss}
class VICRegLoss(nn.Module):
    """
    Computes the VICReg loss proposed in https://arxiv.org/abs/2105.04906.
    Implementation adapted from https://github.com/jolibrain/vicreg-loss/.
    """
    def __init__(
        self,
        var_coeff: float = 1.0,
        inv_coeff: float = 1.0,
        cov_coeff: float = 1e-5,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    # TODO: Look at incorporating the negative sample into VICReg
    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        """
        Calculate the VICReg loss.
        """

        variance_loss = (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma)) / 2

        invariance_loss = self.invariance_loss(x, y)

        covariance_loss = (self.covariance_loss(x) + self.covariance_loss(y)) / 2

        return self.var_coeff * variance_loss + self.inv_coeff * invariance_loss + self.cov_coeff * covariance_loss

    @staticmethod
    def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the invariance loss. Force the representations of the same object to be similar.
        """
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: Tensor, gamma: float) -> Tensor:
        """
        Computes the variance loss. Push the representations across the batch to have high variance.
        """
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: Tensor) -> Tensor:
        """
        Computes the covariance loss. Decorrelate the embeddings' dimensions, pushing the model to capture more information per dimension.
        """
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss


def evaluate(
    encoder: nn.Module,
    test_data: DataLoader,
    loss_fcn: nn.Module,
    epoch: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate the encoder network on the test set.
    """

    encoder.eval()

    validation_losses = {}
    euclidean_distance_ap = 0.0
    euclidean_distance_an = 0.0
    similarity_ap = 0.0
    similarity_an = 0.0

    with torch.no_grad():
        for i, (anchor, positive, negative) in tqdm(enumerate(test_data), f"Evaluation {epoch + 1}", unit="batch", total=len(test_data)):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_features = encoder(anchor)
            positive_features = encoder(positive)
            negative_features = encoder(negative)
            
            batch_losses: Dict[str, Tensor] = loss_fcn(anchor_features, positive_features, negative_features)

            validation_losses = {k: validation_losses.get(k, 0) + v for k, v in batch_losses.items()}

            euclidean_distance_ap += torch.norm(
                anchor_features - positive_features, dim=1
            ).mean()

            euclidean_distance_an += torch.norm(
                anchor_features - negative_features, dim=1
            ).mean()

            similarity_ap += torch.cosine_similarity(
                anchor_features, positive_features
            ).mean()
            similarity_an += torch.cosine_similarity(
                anchor_features, negative_features
            ).mean()

    validation_losses = {k: v / len(test_data) for k, v in validation_losses.items()}

    return {
        **validation_losses,
        "Euclidean Distance Difference": (euclidean_distance_an - euclidean_distance_ap) / len(test_data),
        "Cosine Similarity Difference": (similarity_ap - similarity_an) / len(test_data),
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

        batch_losses: Dict[str, Tensor] = loss_fcn(anchor_features, positive_features, negative_features)

        # Log loss to tensorboard
        logger.log(
            {"Train": {**batch_losses, "Learning Rate": optimizer.param_groups[-1]["lr"]}},
            step=epoch * len(train_data) * train_data.batch_size + i * train_data.batch_size,
        )

        batch_loss = batch_losses["Overall Loss"]
        
        batch_loss.backward()
        optimizer.step()


def train(
    encoder: nn.Module,
    expander: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    loss_fcn: nn.Module,
    epochs: int = 10,
    device: str = "cpu",
    log_dir: str = "./logs",
    output_dir: str = "./models",
    model_name: str = "ConvNeXt-T",
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

    optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, last_epoch=epochs - 1)

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

        metrics_str = " ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1} {metrics_str}")

        logger.log({"Val": metrics}, step=logger.step)

        scheduler.step()

        torch.save(encoder.state_dict(), f"{os.path.join(output_dir, model_name)}/checkpoint-{epoch + 1}.pt")

    logger.finish()


if __name__ == "__main__":
    # Set the seed
    torch.manual_seed(42)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    encoder = models.convnext_tiny(weights="DEFAULT")

    encoder = encoder.to(device)

    expander = nn.Sequential(
            nn.Linear(1000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(True),
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(True),
            nn.Linear(4000, 4000),
    ).to(device)


    # Load the data
    train_transformations = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((256, 192)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomPerspective(interpolation=transforms.InterpolationMode.NEAREST, fill=(0.9536, 0.9470, 0.9417))]),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transformations = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((256, 192)), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    train_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="train_pairs_cropped.txt", transformations=train_transformations)

    test_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="test_pairs_paired_cropped.txt", transformations=test_transformations)

    train_loader = DataLoader(train_data, batch_size=42, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=42, shuffle=False)

    loss_fcn = EncoderLoss(expander, triplet_weight=5.0, vicreg_weight=0.5, margin=1.5)

    train(
        encoder=encoder,
        expander=expander,
        train_data=train_loader,
        test_data=test_loader,
        loss_fcn=loss_fcn,
        epochs=10,  
        device=device,
        model_name="ConvNeXt-T",
    )
