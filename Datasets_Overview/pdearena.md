# PDEArena Datasets
Only focusing on INS/SW. Download instructions can be found in PDEArena's HuggingFace
[Paper](https://arxiv.org/pdf/2209.15616)
[Github](https://github.com/pdearena/pdearena/tree/main)
[HuggingFace](https://huggingface.co/pdearena)

## Shallow Water
Data format: Many files, already split into train/test/valid upon download

In each of train/test/val folders, the data is organized according to 'seed=(NUM)', 

In each 'seed=(NUM)' folder, there are multiple 'runXXXX' folders, each with a singular 'output.nc' file.

Converting the structure above into h5 files, labeled 'seedNUM_runXXXX_output.h5', we get the following 9 members for each .h5 file:

### 'div':
shape (88, 1, 96, 192)

### 'lat':
shape (96,)

### 'lev':
"lev": shape (1,)

### 'lon':
shape (192,)

### 'pres':
shape (88, 96, 192)

### 'time':
"time": shape (88,)

### 'u', 'v', 'vor':
shape (88, 1, 96, 192) for all 3


## Incompressible Navier-Stokes (unconditioned/conditioned)
Data format: Many files, already split into train/test/valid upon download, each .h5 file has 10 members:

### 'buo_y':
shape (sims,)

### 'dt', 'dx', 'dy':
All 3 of shape (sims,)

### 't':
shape (sims, timesteps)

### 'u', 'vx', 'vy': 
All 3 of shape (sims, timesteps, 128, 128)

### 'x', 'y':
shape (sims, 128)






