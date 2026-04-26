from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _train_transform():
    return v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def _eval_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    train_set = datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=_train_transform()
    )
    test_set = datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=_eval_transform()
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
