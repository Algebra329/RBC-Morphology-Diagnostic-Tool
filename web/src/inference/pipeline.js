/**
 * Runs the full inference pipeline client-side via ONNX.js.
 * Reads tensor shapes/normalization from ../../ml/MODEL_IO.md -- keep in sync.
 *
 * Flow:
 *   1. Load segmentation.onnx, run on the full smear image -> bounding boxes
 *   2. Crop each detected region from the source image
 *   3. Load classification.onnx, run on each crop -> class + confidence
 *   4. Return structured results: [{ bbox, class, confidence }, ...]
 */

// TODO: implement once ml/MODEL_IO.md is finalized
export async function runInference(imageElement) {
  throw new Error("not implemented");
}
