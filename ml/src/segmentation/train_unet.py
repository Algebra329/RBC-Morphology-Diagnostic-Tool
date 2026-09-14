"""
Train MONAI U-Net for RBC segmentation.
Input: full smear images + masks from data_prep.py
Output: trained checkpoint + IoU eval
"""

def train():
    """TODO: implement MONAI U-Net training loop."""
    raise NotImplementedError

def evaluate(model, val_loader):
    """TODO: compute IoU on validation set."""
    raise NotImplementedError

if __name__ == "__main__":
    train()
