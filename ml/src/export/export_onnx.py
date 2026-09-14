"""
Export trained segmentation + classification models to ONNX.
Copies .onnx files to ../../web/public/models/
Updates ../MODEL_IO.md with final tensor shapes -- keep this in sync.
"""

def export_segmentation_model(checkpoint_path: str, out_path: str):
    """TODO: torch.onnx.export for the U-Net."""
    raise NotImplementedError

def export_classification_model(checkpoint_path: str, out_path: str):
    """TODO: torch.onnx.export for EfficientNet-B0."""
    raise NotImplementedError

if __name__ == "__main__":
    # TODO: run both exports, copy outputs to web/public/models/
    pass
