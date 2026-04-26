"""Export the trained MNISTNet to ONNX for in-browser inference."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.model import MNISTNet  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "public" / "mnist.onnx"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Run `python -m src.train` first."
        )

    model = MNISTNet()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    size_kb = args.output.stat().st_size / 1024
    print(f"Exported {args.output.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
