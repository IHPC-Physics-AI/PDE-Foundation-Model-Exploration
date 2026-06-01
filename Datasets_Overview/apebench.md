# APEBench Datasets
Only focusing on GSDR. Download instructions can be found in APEBench's Github
[Paper](https://arxiv.org/pdf/2411.00180)
[Github](https://github.com/tum-pbs/apebench/tree/main)
[HuggingFace](https://huggingface.co/datasets/thuerey-group/apebench-scraped)

## Gray-Scott Diffusion-Reaction
Data format: Multiple files, 10 members

### 'norm_const_max', 'norm_const_mean', 'norm_const_min', 'norm_const_std':
Essentially just the feed rate, kill rate, all 4 of shape (2,)

std values are both zero

### 'norm_fields_sca_max', 'norm_fields_sca_mean', 'norm_fields_sca_min', 'norm_fields_sca_std', 'norm_fields_std':
All 5 of shape (2,1,1)

### 'sims':
Varying number of sims depending on variant of GSDR, number of members = number of sims

Shape (timesteps,2,256,256)

