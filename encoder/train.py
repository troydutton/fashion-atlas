import math
import os
import random
from datetime import datetime
from typing import Callable, Dict, Tuple

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
from utils import build_encoder, cosine_distance, get_transforms, pairwise_cosine_distance, parse_config, set_random_seed
from wandb.wandb_run import Run

DRESSCODE_ROOT = "data/DressCode/"

DIRECTORY_MAP = ["upper_body", "lower_body", "dresses"]

class DressCodeDataset(Dataset):
    def __init__(self, root: str, pairs: str, transformations: transforms.Compose) -> None:
        super().__init__()

        self.root = root

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
            margin: float = 1.0,
            initial_temperature: float = 1.0,
            temperature_decay: float = 1.0,
            minimum_temperature: float = 0.0,
        ) -> None:
        super().__init__()

        self.triplet_weight = triplet_weight
        self.vicreg_weight = vicreg_weight
        self.expander = expander

        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=margin)
        self.triplet_loss = BatchHardTripletMarginLoss(margin=margin, initial_temperature=initial_temperature, temperature_decay=temperature_decay, minimum_temperature=minimum_temperature)
        # self.triplet_loss = BatchAllTripletMarginLoss(margin=margin)
        
        self.vicreg_loss = VICRegLoss(var_coeff, inv_coeff, cov_coeff)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Dict[str, Tensor]:
        """
        Calculate the contrastive loss between the anchor, positive and negative samples. 
        Incorporates VICReg loss to prevent information collapse and encourage diversity in the embeddings.

        Returns a dictionary containing the weighted overall loss and unweighted sublosses.
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

class BatchHardTripletMarginLoss(nn.Module):
    def __init__(
            self,
            distance_function: Callable[[Tensor, Tensor], Tensor] = cosine_distance, 
            pairwise_distance_function: Callable[[Tensor, Tensor], Tensor] = pairwise_cosine_distance, 
            margin: float = 1.0,
            initial_temperature: float = 1.0,
            temperature_decay: float = 1.0,
            minimum_temperature: float = 0.0,
        ) -> None:
        super().__init__()

        assert initial_temperature >= 0, "Temperature must be non-negative."
        assert temperature_decay >= 0, "Temperature decay must be non-negative."
        assert minimum_temperature >= 0, "Minimum temperature must be non-negative."

        self.distance_function = distance_function
        self.pairwise_distance_function = pairwise_distance_function
        self.temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.minimum_temperature = minimum_temperature
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """
        Calculate the triplet margin loss between the anchor, positive and negative samples.
        Choose negatives based on the distance between the anchor and the negative
        samples, giving preference to closer (harder) negatives.
        When the temperature is set to 0, the closest negative is always chosen.
        """

        distance_ap = self.distance_function(anchor, positive)
        distance_an = self.pairwise_distance_function(anchor, negative)

        if self.temperature == 0:
            distance_an = distance_an.min(dim=1).values
        else:
            distribution_an = (-distance_an / self.temperature).softmax(dim=1)
            indices_an = torch.multinomial(distribution_an, num_samples=1).squeeze()
            distance_an = distance_an[torch.arange(distance_an.shape[0]), indices_an]

        loss = F.relu(distance_ap - distance_an + self.margin).mean()

        return loss
    
    def decay_temperature(self) -> None:
        self.temperature = max(self.minimum_temperature, self.temperature * self.temperature_decay)

class BatchAllTripletMarginLoss(nn.Module):
    def __init__(
            self,
            distance_function: Callable[[Tensor, Tensor], Tensor] = cosine_distance, 
            pairwise_distance_function: Callable[[Tensor, Tensor], Tensor] = pairwise_cosine_distance, 
            margin: float = 1.0,
        ) -> None:
        super().__init__()

        self.distance_function = distance_function
        self.pairwise_distance_function = pairwise_distance_function
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """
        Calculate the triplet margin loss between the anchor, positive and negative samples.
        Calculates the loss for all possible anchor-negative pairs in the batch.
        """

        distance_ap = self.distance_function(anchor, positive)
        distance_ap = distance_ap.unsqueeze(dim=1).repeat(1, distance_ap.shape[0])
        distance_an: Tensor = self.pairwise_distance_function(anchor, negative)

        loss = F.relu(distance_ap - distance_an + self.margin).mean()

        return loss
    
class CosineAnnealingWarmup:
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, period: int, min_lr: float = 1e-6, max_lr: float = 1e-3) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = period
        self.min_lr = min_lr
        self.max_lr = max_lr 

        self.current_step = 0

        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def step(self) -> None:
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.current_step += 1

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * self.current_step / (self.warmup_steps - 1)
        else:
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.max_steps)))

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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        train_one_epoch(
            encoder=encoder,
            optimizer=optimizer,
            loss_fcn=loss_fcn,
            dataloader=train_loader,
            logger=logger,
            epoch=epoch,
            device=device,
        )

        metrics = evaluate(
            encoder=encoder,
            loss_fcn=loss_fcn,
            dataloader=test_loader,
            epoch=epoch,
            device=device,
        )

        metrics_str = " ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1} {metrics_str}")

        logger.log({"Val": metrics}, step=logger.step)

        scheduler.step()

        torch.save(encoder.state_dict(), f"{os.path.join(output_dir, run_name)}/checkpoint-{epoch + 1}.pt")

    logger.finish()

def train_one_epoch(
    encoder: nn.Module,
    optimizer: optim.Optimizer,
    loss_fcn: nn.Module,
    dataloader: DataLoader,
    logger: Run,
    epoch: int,
    device: str = "cpu",
):
    """
    Train the encoder network for one epoch.
    """

    encoder.train()

    euclidean_distance_ap = 0.0
    euclidean_distance_an = 0.0
    hard_euclidean_distance_an = 0.0
    cosine_distance_ap = 0.0
    cosine_distance_an = 0.0
    hard_cosine_distance_an = 0.0

    for i, (anchor, positive, negative) in tqdm(enumerate(dataloader), f"Training (Epoch {epoch + 1})", unit="batch", total=len(dataloader)):
        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_features = encoder(anchor)
        positive_features = encoder(positive)
        negative_features = encoder(negative)

        batch_losses: Dict[str, Tensor] = loss_fcn(anchor_features, positive_features, negative_features)

        euclidean_distance_ap += torch.norm(anchor_features - positive_features, dim=1).mean()
        euclidean_distance_an += torch.norm(anchor_features - negative_features, dim=1).mean()
        hard_euclidean_distance_an += torch.cdist(anchor_features, negative_features).min(dim=1).values.mean()

        cosine_distance_ap += cosine_distance(anchor_features, positive_features).mean()
        cosine_distance_an += cosine_distance(anchor_features, negative_features).mean()
        hard_cosine_distance_an += pairwise_cosine_distance(anchor_features, negative_features).min(dim=1).values.mean()

        # TODO: Find a better way to log & step the temperature
        # Log loss to tensorboard
        logger.log(
            {"Train": {**batch_losses, "Learning Rate": optimizer.param_groups[-1]["lr"], "Temperature": loss_fcn.triplet_loss.temperature}},
            step=epoch * len(dataloader) * dataloader.batch_size + i * dataloader.batch_size,
        )

        loss_fcn.triplet_loss.decay_temperature()

        batch_loss = batch_losses["Overall Loss"]
        
        batch_loss.backward()
        optimizer.step()

    logger.log(
        {"Train": {
            "Euclidean Distance Difference": (euclidean_distance_an - euclidean_distance_ap) / len(dataloader),
            "Hard Euclidean Distance Difference": (hard_euclidean_distance_an - euclidean_distance_ap) / len(dataloader),
            "Cosine Similarity Difference": (cosine_distance_an - cosine_distance_ap) / len(dataloader),
            "Hard Cosine Similarity Difference": (hard_cosine_distance_an - cosine_distance_ap) / len(dataloader)}},
        step=logger.step
    )

def evaluate(
    encoder: nn.Module,
    loss_fcn: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate the encoder network. Returns a dictionary of metrics.
    """

    encoder.eval()

    validation_losses = {}
    euclidean_distance_ap = 0.0
    euclidean_distance_an = 0.0
    hard_euclidean_distance_an = 0.0
    cosine_distance_ap = 0.0
    cosine_distance_an = 0.0
    hard_cosine_distance_an = 0.0

    all_anchor_features = []
    all_positive_features = []

    with torch.no_grad():
        for _, (anchor, positive, negative) in tqdm(enumerate(dataloader), f"Evaluation (Epoch {epoch + 1})", unit="batch", total=len(dataloader)):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_features = encoder(anchor)
            positive_features = encoder(positive)
            negative_features = encoder(negative)

            all_anchor_features.append(anchor_features)
            all_positive_features.append(positive_features)
            
            batch_losses: Dict[str, Tensor] = loss_fcn(anchor_features, positive_features, negative_features)

            validation_losses = {k: validation_losses.get(k, 0) + v for k, v in batch_losses.items()}

            euclidean_distance_ap += torch.norm(anchor_features - positive_features, dim=1).mean()
            euclidean_distance_an += torch.norm(anchor_features - negative_features, dim=1).mean()
            hard_euclidean_distance_an += torch.cdist(anchor_features, negative_features).min(dim=1).values.mean()

            cosine_distance_ap += cosine_distance(anchor_features, positive_features).mean()
            cosine_distance_an += cosine_distance(anchor_features, negative_features).mean()
            hard_cosine_distance_an += pairwise_cosine_distance(anchor_features, negative_features).min(dim=1).values.mean()

    all_anchor_features = torch.cat(all_anchor_features)
    all_positive_features = torch.cat(all_positive_features)

    all_cosine_distance = pairwise_cosine_distance(all_anchor_features, all_positive_features)

    accuracy = (all_cosine_distance.argmin(dim=1) == torch.arange(len(all_cosine_distance), device=device)).count_nonzero() / len(all_cosine_distance)

    validation_losses = {k: v / len(dataloader) for k, v in validation_losses.items()}

    return {
        **validation_losses,
        "Accuracy": accuracy.item(),
        "Euclidean Distance Difference": (euclidean_distance_an - euclidean_distance_ap) / len(dataloader),
        "Hard Euclidean Distance Difference": (hard_euclidean_distance_an - euclidean_distance_ap) / len(dataloader),
        "Cosine Similarity Difference": (cosine_distance_an - cosine_distance_ap) / len(dataloader),
        "Hard Cosine Similarity Difference": (hard_cosine_distance_an - cosine_distance_ap) / len(dataloader),
    }

if __name__ == "__main__":
    set_random_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_config("config/DressCode.yaml")

    # Load data
    train_transformations, test_transformations = get_transforms()

    train_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="train_pairs_cropped.txt", transformations=train_transformations)

    test_data = DressCodeDataset(root=DRESSCODE_ROOT, pairs="test_pairs_paired_cropped.txt", transformations=test_transformations)
    
    # Instantiate the model
    encoder, expander = build_encoder(**args["model"], device=device)
    
    # Create the optimizer, scheduler and loss function
    optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()))

    scheduler = CosineAnnealingWarmup(optimizer, **args["scheduler"])

    loss_fcn = EncoderLoss(expander, **args["loss"])

    # Start logging
    os.makedirs("./logs", exist_ok=True)

    logger = wandb.init(dir="./logs", project="fashion-atlas", name=args["train"]["run_name"], config={**args, "train_transformations": train_transformations.__str__(), "test_transformations": test_transformations.__str__()})

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
