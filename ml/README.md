# ML — RBC Morphology Pipeline

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline order
1. `src/data_prep.py` — download/parse Chula-RBC-12 annotations into segmentation masks + classification crops. Outputs to `data/processed/`.
2. `src/segmentation/train_unet.py` — trains the MONAI U-Net segmentation model.
3. `src/classification/train_efficientnet.py` — trains EfficientNet-B0 classifier on cropped cells.
4. `src/export/export_onnx.py` — exports both trained models to ONNX, writes `MODEL_IO.md` with final tensor shapes, copies `.onnx` files to `../web/public/models/`.

## Checkpoint (week 2)
If segmentation isn't reliably isolating cells by the end of week 2, stop and fall back to classification-only (skip step 2, feed pre-cropped cells directly from `data_prep.py` into step 3). See `docs/PRD.md` §8 for the full fallback plan.

## Evaluation
- Segmentation: IoU
- Classification: per-class F1 (accuracy is not a meaningful metric here — 34.5:1 class imbalance)

## Data
Raw Chula-RBC-12 files go in `data/raw/` (gitignored — do not commit dataset files). Processed masks/crops go in `data/processed/` (also gitignored).
