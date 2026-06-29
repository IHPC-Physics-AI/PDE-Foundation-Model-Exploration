import sys
import os

sys.argv = [
    "infer_MORPH.py",
    "--model_choice", "FM",
    "--model_size", "S",
    "--checkpoint", "morph-S-FM-max_ar1_ep225.pth",
    "--test_dataset", "GSDR2D",
    "--ar_order", "1",
    "--rollout_horizon", "10",
    "--batch_size", "1",
    "--test_sample", "0",
    "--max_ar_order", "1"
]
