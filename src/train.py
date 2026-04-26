import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import MNISTNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    with torch.set_grad_enabled(is_train):
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_seen += batch_size
            pbar.set_postfix(loss=f"{total_loss / total_seen:.4f}",
                             acc=f"{total_correct / total_seen:.4f}")

    return total_loss / total_seen, total_correct / total_seen


def plot_curves(history, path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MNISTNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_path = CHECKPOINT_DIR / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  ↳ saved new best to {best_path} (val_acc={val_acc:.4f})")

    plot_curves(history, OUTPUT_DIR / "training_curves.png")
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Curves saved to {OUTPUT_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
