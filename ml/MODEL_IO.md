# Model I/O Contract

Filled in by the ML side once training is finalized. The web side integrates against this file, not the training code.

## `segmentation.onnx`

- **Input tensor shape:** TBD e.g. `[1, 3, H, W]`
- **Input dtype / normalization:** TBD (e.g. float32, mean/std used)
- **Output:** TBD — list of bounding boxes `[x, y, w, h]` or a mask tensor; specify which
- **Notes:** any preprocessing the web side must replicate exactly (resize method, color order RGB vs BGR, etc.)

## `classification.onnx`

- **Input tensor shape:** TBD e.g. `[1, 3, 224, 224]` (single cell crop)
- **Input dtype / normalization:** TBD
- **Output:** 12-length probability vector
- **Class label order (index → class):**
  0. Normal
  1. Spherocyte
  2. Target Cell
  3. Ovalocyte
  4. Stomatocyte
  5. Hypochromia
  6. Schistocyte
  7. Burr Cell
  8. Macrocyte
  9. Microcyte
  10. Teardrop
  11. Uncategorized

## Last updated
Fill in date + who updated when this file changes. If the shape changes, ping the web side before merging.
