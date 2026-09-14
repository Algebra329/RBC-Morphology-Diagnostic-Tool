"""
Parse Chula-RBC-12 annotations into:
  1. Segmentation masks (for U-Net training)
  2. Single-cell classification crops (for EfficientNet training)

Outputs to ../data/processed/{masks,crops}/
"""

def load_annotations(raw_dir: str):
    """Load Chula-RBC-12 point annotations. TODO: implement."""
    raise NotImplementedError

def make_segmentation_masks(annotations, out_dir: str):
    """Generate fixed-radius masks per labeled cell for U-Net training. TODO: implement."""
    raise NotImplementedError

def make_classification_crops(annotations, out_dir: str):
    """Crop single-cell patches per class for the classifier. TODO: implement."""
    raise NotImplementedError

def report_class_distribution(annotations):
    """Print/plot the class imbalance (expect ~34.5:1 majority:minority). TODO: implement."""
    raise NotImplementedError

if __name__ == "__main__":
    # TODO: wire up CLI args (raw data path, output path)
    pass
