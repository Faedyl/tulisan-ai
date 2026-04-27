# Pendeteksian Tulisan Tangan — MNIST

A convolutional neural network trained on MNIST for handwritten digit
classification, served as a fully in-browser inference demo.

> **Kuliah · Deep Learning · Tugas 01**
> Test accuracy: **99.46 %** on the MNIST test split (10,000 images).

---

## Demo

The deployed site renders a 280 × 280 drawing canvas. Strokes are bounding-box
cropped, scaled so the longest side spans 20 pixels, and re-centred within a
28 × 28 frame — matching MNIST's preprocessing convention. The forward pass
runs entirely on the client through WebAssembly via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). No image data
leaves the device.

Typical inference latency: **1–10 ms** in modern browsers.

---

## Project layout

```
.
├── public/                 # Static site deployed to Vercel
│   ├── index.html          # Drawing canvas + onnxruntime-web inference
│   ├── mnist.onnx          # Exported model graph
│   └── mnist.onnx.data     # External weights (~559 KB)
├── src/
│   ├── model.py            # MNISTNet architecture
│   ├── dataset.py          # MNIST dataloaders + normalisation
│   ├── train.py            # Training loop (AdamW + OneCycleLR)
│   ├── evaluate.py         # Test accuracy + confusion matrix
│   └── server.py           # (legacy) FastAPI server — see "Why static?" below
├── scripts/
│   └── export_onnx.py      # Convert checkpoints/best.pt → public/mnist.onnx
├── checkpoints/best.pt     # Trained weights (gitignored)
├── outputs/                # Training curves + confusion matrix (gitignored)
├── requirements.txt
└── vercel.json             # Static-only deploy config
```

---

## Model

| Field                  | Value                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Architecture           | 3 conv blocks · 32 → 64 → 128 channels · BatchNorm + Dropout (0.25) · GAP · linear head  |
| Trainable parameters   | 119,498                                                                                  |
| Optimiser              | AdamW · lr 1 × 10⁻³ · weight decay 1 × 10⁻⁴                                              |
| Schedule               | OneCycleLR · 15 epochs · label smoothing 0.05                                            |
| Augmentation           | RandomAffine ±10° · translate ±10 %                                                      |
| Test accuracy          | **99.46 %**                                                                              |
| Inference runtime      | onnxruntime-web · WebAssembly · client-side                                              |

---

## Reproducing

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train

```bash
python -m src.train --epochs 15 --batch-size 128
```

This downloads MNIST into `data/`, trains the CNN, and writes the best
checkpoint to `checkpoints/best.pt`. Training curves are saved to
`outputs/training_curves.png`.

### 3. Evaluate

```bash
python -m src.evaluate
```

Prints test accuracy + per-class metrics and saves
`outputs/confusion_matrix.png`.

### 4. Export to ONNX

```bash
python -m scripts.export_onnx
```

Writes `public/mnist.onnx` (graph) and `public/mnist.onnx.data` (weights).
This is the only artefact the deployed site needs.

### 5. Run locally

The frontend is a single static HTML file. Any static server works:

```bash
cd public && python -m http.server 8000
# open http://localhost:8000
```

---

## Why static? (a.k.a. the Vercel 500 MB story)

The original plan was to deploy `src/server.py` — a FastAPI service that loads
the PyTorch checkpoint and serves `/api/predict` — as a Vercel Serverless
Function. **This blew through the 250 MB unzipped / 500 MB total Lambda size
limit on the first deploy**: the `torch` wheel alone is ~700 MB.

Things that were tried before giving up on the server route:

1. **Pinning `torch` to the CPU-only wheel** (`torch==2.2.x+cpu`) — still
   ~180 MB unzipped, plus `torchvision`, plus `fastapi`, plus transitive deps.
   Still over.
2. **Stripping `torchvision`** and reimplementing the v2 transforms with
   `Pillow` + NumPy. Saved a few MB. Still over.
3. **Switching the runtime to `onnxruntime` (CPU) on the server** so PyTorch
   could be removed entirely. Closer, but cold starts on a 200 MB+ Lambda
   were 8–15 s — unusable for a demo where the user expects a result the
   moment they finish drawing a digit.

**The fix was to stop running inference on the server at all.** The model is
exported to ONNX once (~559 KB of weights), shipped as a static asset, and
executed in the browser via `onnxruntime-web`. The deployment is now:

- A handful of static files in `public/` — no Lambda involved.
- `vercel.json` sets `outputDirectory: "public"` and adds long-cache headers
  for the `.onnx` blob.
- Cold starts are gone. Inference is faster (1–10 ms) than the round-trip
  to a server would have been.
- As a bonus: nothing the user draws ever leaves their device.

`src/server.py` is kept in the repo for reference and for local development
(`python -m src.server` still works), but it is not part of the deployment.

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
   Gradient-based learning applied to document recognition.
   *Proceedings of the IEEE, 86*(11), 2278–2324.
2. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
   *International Conference on Learning Representations*.
3. Smith, L. N., & Topin, N. (2018). Super-convergence: very fast training of
   neural networks using large learning rates. *arXiv:1708.07120*.
