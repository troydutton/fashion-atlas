import os
import random
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import wandb
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import build_model, get_transforms, parse_config, set_random_seed
from wandb.wandb_run import Run

DRESSCODE_ROOT = "data/DressCode/"

DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

class DressCodeDataset(Dataset):
    def __init__(self, root: str, pairs: str, transformations: transforms.Compose) -> None:
        super().__init__()

        # Root directory of the dataset
        self.root = root

        # Model-garment pairs
        self.data = pd.read_csv(os.path.join(self.root, pairs), delimiter="\t", header=None, names=["model", "garment", "label"])

        self.transformations = transformations

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        model, garment, label = self.data.iloc[index]

        model_path = os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", model)

        garment_path = os.path.join(self.root, DIRECTORY_MAP[label], "cropped_images", garment)

        # Load the anchor & positive images
        if random.choice([True, False]):
            anchor = Image.open(model_path).convert("RGB")

            positive = Image.open(garment_path).convert("RGB")
        else:
            anchor = Image.open(garment_path).convert("RGB")

            positive = Image.open(model_path).convert("RGB")

        # Randomly sample a negative (ensuring it is not the same as the anchor)
        while (negative_index := random.randrange(0, len(self.data))) == index:
            pass

        negative_model, negative_garment, negative_label = self.data.iloc[negative_index]

        # Load the negative image
        if random.choice([True, False]):
            negative = Image.open(os.path.join(self.root, DIRECTORY_MAP[negative_label], "cropped_images", negative_garment,)).convert("RGB")
        else:
            negative = Image.open(os.path.join(self.root, DIRECTORY_MAP[negative_label], "cropped_images", negative_model)).convert("RGB")

        anchor = self.transformations(anchor)
        positive = self.transformations(positive)
        negative = self.transformations(negative)

        return anchor, positive, negative
class EncoderLoss(nn.Module):
    def __init__(
            self, 
            expander: nn.Module, 
            triplet_weight: float = 1.0, 
            vicreg_weight: float = 1.0, 
            var_coeff: float = 1.0, 
            inv_coeff: float = 1.0, 
            cov_coeff: float=1e-5, 
            margin: float = 1.0
        ) -> None:
        super().__init__()

        self.triplet_weight = triplet_weight
        self.vicreg_weight = vicreg_weight
        self.expander = expander

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y),
            margin=margin,
        )
        
        self.vicreg_loss = VICRegLoss(var_coeff, inv_coeff, cov_coeff)

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
    def variance_loss(x: Tensor, gamma: float) -> Tensor:
        """
        Computes the variance loss. Push the representations across the batch to have high variance.
        """
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss
    
    @staticmethod
    def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the invariance loss. Force the representations of the same object to be similar.
        """
        return F.mse_loss(x, y)

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
    loss_fcn: nn.Module,
    test_loader: DataLoader,
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
        for i, (anchor, positive, negative) in tqdm(enumerate(test_loader), f"Evaluation {epoch + 1}", unit="batch", total=len(test_loader)):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_features = encoder(anchor)
            positive_features = encoder(positive)
            negative_features = encoder(negative)
            
            batch_losses: Dict[str, Tensor] = loss_fcn(anchor_features, positive_features, negative_features)

            validation_losses = {k: validation_losses.get(k, 0) + v for k, v in batch_losses.items()}

            euclidean_distance_ap += torch.norm(anchor_features - positive_features, dim=1).mean()
            euclidean_distance_an += torch.norm(anchor_features - negative_features, dim=1).mean()

            similarity_ap += torch.cosine_similarity(anchor_features, positive_features).mean()
            similarity_an += torch.cosine_similarity(anchor_features, negative_features).mean()

    validation_losses = {k: v / len(test_loader) for k, v in validation_losses.items()}

    return {
        **validation_losses,
        "Euclidean Distance Difference": (euclidean_distance_an - euclidean_distance_ap) / len(test_loader),
        "Cosine Similarity Difference": (similarity_ap - similarity_an) / len(test_loader),
    }

def train_one_epoch(
    encoder: nn.Module,
    optimizer: optim.Optimizer,
    loss_fcn: nn.Module,
    train_loader: DataLoader,
    logger: Run,
    epoch: int,
    device: str = "cpu",
):
    """
    Train the encoder network for one epoch.
    """

    encoder.train()

    for i, (anchor, positive, negative) in tqdm(enumerate(train_loader), f"Training {epoch + 1}", unit="batch", total=len(train_loader)):
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
            step=epoch * len(train_loader) * train_loader.batch_size + i * train_loader.batch_size,
        )

        batch_loss = batch_losses["Overall Loss"]
        
        batch_loss.backward()
        optimizer.step()

def train(
    encoder: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    train_data: Dataset,
    test_data: Dataset,
    loss_fcn: nn.Module,
    logger: Run,
    epochs: int = 10,
    batch_size: int = 42,
    device: str = "cpu",
    output_dir: str = "./models",
    run_name: str = "ConvNeXt-T",
):
    """
    Train the encoder network for a set number of epochs.
    """
    run_name = f"{run_name} {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, run_name), exist_ok=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        train_one_epoch(
            encoder=encoder,
            optimizer=optimizer,
            loss_fcn=loss_fcn,
            train_loader=train_loader,
            logger=logger,
            epoch=epoch,
            device=device,
        )

        metrics = evaluate(
            encoder=encoder,
            loss_fcn=loss_fcn,
            test_loader=test_loader,
            epoch=epoch,
            device=device,
        )

        metrics_str = " ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1} {metrics_str}")

        logger.log({"Val": metrics}, step=logger.step)

        scheduler.step()

        torch.save(encoder.state_dict(), f"{os.path.join(output_dir, run_name)}/checkpoint-{epoch + 1}.pt")

    logger.finish()


if __name__ == "__main__":
    set_random_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_config("config/DressCode.yaml")

    # Load data
    train_transformations, test_transformations = get_transforms()

    train_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="train_pairs_cropped.txt", transformations=train_transformations)

    test_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="test_pairs_paired_cropped.txt", transformations=test_transformations)
    
    # Load the model and optimizer
    encoder, expander = build_model(**args["model"], device=device)

    optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()), **args["optimizer"])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    loss_fcn = EncoderLoss(expander, **args["loss"])

    # Start logging
    os.makedirs("./logs", exist_ok=True)

    logger = wandb.init(dir="./logs", project="fashion-atlas", name=args["train"]["run_name"], config=args)

    # Train the model
    train(
        encoder=encoder,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fcn=loss_fcn,
        train_data=train_data,
        test_data=test_data,
        logger=logger,
        device=device,
        **args["train"],
    )
