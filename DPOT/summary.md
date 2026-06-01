# Summary for DPOT
Relatively simple model to work with. Pretrained weights can be downloaded from DPOT's Github.

General pipeline: Preprocess data using DPOT's script -> Run inference using DPOT's script (evaluate.py)

For reference, check the notebooks in this folder.

## Notes
Remember to change file paths in make_master_file.py.

debug.py is a script used for debugging, as the name implies.

evaluate.py scripts include custom plotting code, which were not included in DPOT's original evaluate.py script.

## evaluate.py VS evaluate_unknown.py
Dataset used: PDEBench 2D Diffusion-Reaction

evaluate.py: By default, takes in timestep 0-9 to autoregressively predict the remaining timesteps, until timestep 100.

evaluate_unknown.py: Takes in timestep 91-100 to autoregressively predict timesteps 101 to 110, in which the ground truth does not exist in the data.