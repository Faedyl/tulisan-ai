import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import v2

from src.dataset import MNIST_MEAN, MNIST_STD
from src.model import MNISTNet
from src.train import pick_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"
WEB_DIR = Path(__file__).resolve().parent / "web"

DEVICE = pick_device()

_preprocess = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MNIST_MEAN, MNIST_STD),
])


def _load_model() -> MNISTNet:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"No checkpoint at {CHECKPOINT_PATH}. Run `python -m src.train` first."
        )
    model = MNISTNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


MODEL = _load_model()
N_PARAMS = sum(p.numel() for p in MODEL.parameters())


def _decode_image(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.asarray(img)


def _normalize_to_mnist(arr: np.ndarray) -> Image.Image | None:
    # Frontend canvas has a transparent background and black ink strokes,
    # so the alpha channel doubles as the ink mask (0 = empty, 255 = stroke)
    # — this matches MNIST's convention of white digit on black background.
    if arr.ndim == 3 and arr.shape[2] == 4:
        ink = arr[..., 3]
    else:
        gray = arr[..., :3].astype(np.float32).mean(axis=2)
        ink = (255.0 - gray).clip(0, 255).astype(np.uint8)

    mask = ink > 30
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = Image.fromarray(crop, mode="L").resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
    return canvas


class PredictRequest(BaseModel):
    image: str


app = FastAPI(title="MNIST Lab")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/meta")
def meta():
    return {"params": N_PARAMS, "device": str(DEVICE)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    arr = _decode_image(req.image)
    img = _normalize_to_mnist(arr)
    if img is None:
        return {"empty": True, "predictions": [], "inference_ms": 0.0}

    tensor = _preprocess(img).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    inference_ms = (time.perf_counter() - t0) * 1000.0

    predictions = sorted(
        [{"label": i, "prob": float(probs[i])} for i in range(10)],
        key=lambda x: x["prob"],
        reverse=True,
    )
    return {
        "empty": False,
        "predictions": predictions,
        "inference_ms": round(inference_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
