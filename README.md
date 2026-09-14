# RBC Morphology Diagnostic Tool

Offline, browser-based classification of red blood cell morphology from smear images. Built for the UnivaBio hackathon (BioCatalysis × UnivaDev).

Full requirements: [`docs/PRD.md`](docs/PRD.md)

## Architecture

```
smear image → [ml/] segmentation (U-Net) → classification (EfficientNet-B0) → ONNX export
                                                              ↓
                                          [web/] ONNX.js runs both models client-side → UI
```

No server-side inference. No patient image ever leaves the browser.

## Repo layout

- **`ml/`** — data prep, model training, ONNX export. Owner: ML.
- **`web/`** — web app that loads the exported ONNX models and runs inference in-browser. Owner: backend/infra.
- **`docs/`** — PRD, architecture diagram, one-page submission PDF.

## Handoff contract

The ML side delivers two files to `web/public/models/`:
- `segmentation.onnx`
- `classification.onnx`

...plus `ml/MODEL_IO.md`, which documents exact input/output tensor shapes so the web side can integrate without reading training code. **Do not change model I/O shape without updating that file.**

## Getting started

See `ml/README.md` and `web/README.md` for setup in each half.
