# Web — In-Browser Inference App

Loads `segmentation.onnx` and `classification.onnx` (from `public/models/`) and runs both entirely client-side via ONNX.js. No backend inference server.

## Setup
```bash
npm install
npm run dev
```

## Structure
- `src/pages/` — app screens (upload → result view)
- `src/components/` — UI pieces (image upload, bounding-box overlay, result cards, loading animation)
- `src/inference/` — ONNX.js loading + pipeline glue (run segmentation → crop detected regions → run classification → assemble results)

## Integration contract
Read `../ml/MODEL_IO.md` before writing `src/inference/` code — it specifies exact input tensor shape, normalization, and output format for both models. Do not guess these from the `.onnx` files directly; the ML side keeps that doc authoritative.

## Design notes
- Clinical palette: deep blues, whites, soft teals (see `docs/PRD.md` for full design intent).
- Two-stage inference has real latency — the loading state should feel intentional, not like a stall.
- Must work fully offline once loaded (static assets + ONNX models cached).
