# PRD: RBC Morphology Diagnostic Tool
**UnivaBio Hackathon (BioCatalysis × UnivaDev) — Devpost**
Submission deadline: **October 6, 2026**
Team: 2 — ML/model owner + backend/infra owner

---

## 1. Problem Statement

In remote and low-resource regions, there is a severe shortage of trained hematologists able to manually examine peripheral blood smears under a microscope. Diagnosing conditions like Sickle Cell Disease, Thalassemia, and various anemias depends on recognizing red blood cell (RBC) morphology — a skill that takes years to train and isn't available at the point of care in rural clinics.

**Goal:** Build a tool that takes an image of a blood smear and returns per-cell morphological classification (12 classes), running entirely in-browser with no server-side inference — so it works offline, on cheap hardware, without transmitting patient images anywhere.

## 2. Target User

A rural health worker or community clinician with a smartphone or basic laptop, a digital microscope camera or phone-to-eyepiece adapter, and unreliable/no internet access. They are not a data scientist — the tool must be usable with zero ML knowledge.

## 3. Clinical Scope

**Dataset:** Chula-RBC-12 (Chulalongkorn University) — 706 high-resolution smear images, ~20,875 labeled cells, 12 classes. Severe class imbalance (~34.5:1 majority:minority).

**Classes:** Normal, Spherocyte, Target Cell, Ovalocyte, Stomatocyte, Hypochromia, Schistocyte, Burr Cell, Macrocyte, Microcyte, Teardrop, Uncategorized.

**Out of scope for v1:** Diagnosis or treatment recommendations. The tool classifies cell morphology only — it does not output a disease diagnosis. All UI copy must reflect this (informational tool, not a diagnostic device, consult a professional).

## 4. System Architecture

```
[Smear image] 
     ↓
[Segmentation model: U-Net, MONAI-trained]  → isolates individual cells, outputs bounding boxes/masks
     ↓
[Classification model: EfficientNet-B0]     → per-cell class + confidence
     ↓
[Client-side inference: ONNX.js, in-browser]
     ↓
[Web UI: image upload → annotated result]
```

**Key architectural decision:** Both models run client-side via ONNX.js. No image or patient data is ever sent to a server. This is the core "accessible, offline, private" pitch — do not compromise it by moving inference server-side.

---

## 5. Workstream A — ML / Model (Owner: you)

### A1. Data Pipeline
- Parse Chula-RBC-12 point annotations into:
  - Segmentation masks (fixed-radius region per labeled cell) for U-Net training
  - Cropped single-cell patches (labeled by class) for classifier training
- Produce and document class distribution / imbalance stats before training.

### A2. Segmentation Model
- Architecture: U-Net, trained via MONAI.
- Input: full smear image. Output: per-cell mask/bounding box.
- Evaluation metric: IoU.
- **Checkpoint (end of week 2):** if segmentation isn't reliably isolating cells, fall back to a classifier-only pipeline (see §8, Risk & Fallback).

### A3. Classification Model
- Architecture: EfficientNet-B0, transfer learning.
- Training: class-weighted loss + Focal Loss to handle imbalance.
- Augmentation: flips and rotations ONLY. No scaling/resizing — cell size is diagnostic and must not be distorted.
- Evaluation metric: per-class F1 (not accuracy — stated explicitly in submission materials given the imbalance).

### A4. Pipeline Integration
- Full smear → segmentation → per-cell crops → classification → structured output (list of `{bbox, class, confidence}`).
- Must handle segmentation misses/over-splits without crashing (return partial results, not an error state).

### A5. Export
- Export both models to ONNX.
- Deliverable to Workstream B: two `.onnx` files + a documented I/O contract (see §7).

## 6. Workstream B — Backend / Infra (Owner: teammate)

Since inference is fully client-side, "backend" here means infra, delivery, and any optional persistence — not a model-serving API. Suggested scope:

### B1. Frontend/Web App Shell
- Static web app (React/Next.js + TailwindCSS per design guidance below) that:
  - Accepts image upload (or webcam capture)
  - Runs both ONNX models client-side via ONNX.js in sequence
  - Renders bounding boxes + class labels + confidence over the image
  - Shows a real loading/analyzing animation during inference (2-stage model has non-trivial latency — make the wait itself feel like a feature)

### B2. Hosting & Delivery
- Deploy as a static site (e.g., Vercel/Netlify/GitHub Pages) — no backend server required for the core flow.
- Ensure ONNX model files are served efficiently (consider model size vs load time; discuss quantization needs with Workstream A if load time is an issue).

### B3. Optional (only if time allows, does not block submission)
- Local-only history of past analyses (client-side storage, e.g., IndexedDB — **not** a server database, to preserve the privacy pitch).
- Simple export/share of a result (e.g., downloadable report image).

### B4. Repo & Docs
- GitHub repo structure, README with architecture diagram (segmentation → classification → UI), one-click run instructions (`npm install && npm run dev` level of simplicity).
- One-page PDF project description (Devpost requirement).

---

## 7. Interface Contract (A ↔ B handoff)

**Segmentation model (`segmentation.onnx`)**
- Input: image tensor, shape/normalization TBD by Workstream A once finalized — **A must document exact shape, dtype, and normalization before handoff.**
- Output: list of bounding boxes (or mask), one per detected cell.

**Classification model (`classification.onnx`)**
- Input: single-cell image crop, shape/normalization TBD by Workstream A.
- Output: 12-class probability vector.

**Action item:** Workstream A delivers a short `MODEL_IO.md` alongside the `.onnx` files specifying exact tensor shapes, preprocessing steps, and class label order, so Workstream B can integrate without needing to read training code.

---

## 8. Risks & Fallback Plan

| Risk | Fallback |
|---|---|
| Segmentation model unreliable by week 2 checkpoint | Drop segmentation; use pre-cropped single-cell classification only (dataset already supports this via point annotations) |
| ONNX model too large / slow to load in-browser | Investigate quantization; if still an issue, reduce to EfficientNet-B0 only (already smallest viable option) |
| Two-stage pipeline not integrated in time | Ship classification-only as the submitted version; segmentation becomes a "future work" talking point in the video, not a blocker |

## 9. Milestones (against Oct 6 deadline)

- **Week 1:** Data pipeline (A1) + web app shell (B1) in parallel.
- **Week 2:** Segmentation model (A2) + hosting setup (B2). **Checkpoint: go/no-go on segmentation.**
- **Week 3:** Classification model (A3).
- **Week 4:** Pipeline integration (A4) + ONNX export (A5) + frontend integration of both models.
- **Week 5:** UI polish, README, one-page PDF.
- **Week 6 (buffer):** Demo video, bug fixes, submission. No new features after this point.

## 10. Judging Alignment (Devpost criteria)

- **Idea & Innovation:** targeted, specific clinical problem (not a generic health app).
- **Implementation:** real two-stage pipeline, F1-focused evaluation on a genuinely imbalanced dataset.
- **Health Impact & Rigor:** addresses point-of-care access gap; explicit non-diagnostic framing for safety/rigor.
- **Design & Usability:** polished web UI, clear result presentation, works on low-end hardware.
- **Presentation:** README + architecture diagram + one-page PDF + video, per submission requirements.
- Note: rules state judges may ask to be walked through the code — both workstreams should be able to explain their half end-to-end, not just demo it.
