# CFDBench Datasets
Only focusing on INS/SW. Download instructions can be found in CFDBench's HuggingFace
[Paper](https://arxiv.org/pdf/2310.05963)
[Github](https://github.com/luo-yining/CFDBench?tab=readme-ov-file)
[HuggingFace](https://huggingface.co/datasets/chen-yingfa/CFDBench/tree/main)

## Incompressible Navier-Stokes
Split into 4 different flows: Cavity/Tube/Cylinder/Dam

Data format: 


### 'u.npy', 'v.npy':
Shape (timesteps,64,64)

### 'case.json':
"height": 0.01,

"width": 0.01,

"vel_top": 1,

"density": 1,

"viscosity": 1e-05,

"rotated": false

*Values given above are just examples*