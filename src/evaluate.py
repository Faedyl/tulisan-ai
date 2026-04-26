from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_dataloaders
from src.model import MNISTNet
from src.train import pick_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"No checkpoint at {CHECKPOINT_PATH}. Run `python -m src.train` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    _, test_loader = get_dataloaders(batch_size=256, num_workers=2)

    model = MNISTNet().to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    acc = (y_pred == y_true).mean()
    print(f"Test accuracy: {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={acc:.4f})")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
