import sys
import os

# 1. Define the arguments exactly as they appeared in your notebook cell.
# sys.argv[0] is always the script name.
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

# 2. Run the code logic.
# Since infer_MORPH.py executes its logic at the top level (not inside a main() function),
# simply importing it will trigger the script's execution.
try:
    print("Starting debug session...")
    import infer_MORPH
    print("Inference completed successfully.")
except Exception as e:
    print(f"Debugger caught an error: {e}")
    raise