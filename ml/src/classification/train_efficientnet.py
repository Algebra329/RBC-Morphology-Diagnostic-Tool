"""
Fine-tune EfficientNet-B0 on cropped RBC cells (12 classes).
Uses class-weighted loss + Focal Loss for the 34.5:1 imbalance.
Augmentation: flips/rotations ONLY -- never scaling (cell size is diagnostic).
"""

def train():
    """TODO: implement training loop with Focal Loss."""
    raise NotImplementedError

def evaluate(model, val_loader):
    """TODO: compute per-class F1 (primary metric, not accuracy)."""
    raise NotImplementedError

if __name__ == "__main__":
    train()
