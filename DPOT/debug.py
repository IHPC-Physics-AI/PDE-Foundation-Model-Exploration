import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, "DPOT")

sys.path.append(current_dir)
sys.path.append(code_dir)

sys.argv = [
    "evaluate_unknown.py",
    "--model", "DPOT",
    "--dataset", "dr_pdb",
    "--train_paths", "dr_pdb",
    "--test_paths", "dr_pdb",
    "--resume_path", "DPOT/model_S.pth",
    "--n_channels", "4",
    "--T_in", "10",
    "--res", "128",
    "--batch_size", "1",
    "--ntest_list", "5",
    "--gpu", "cpu",
]

if __name__ == "__main__":
    import evaluate_unknown
