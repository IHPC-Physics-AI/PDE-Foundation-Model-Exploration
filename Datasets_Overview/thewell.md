# TheWell Datasets
Only focusing on GSDR. Download instructions can be found in TheWell's Github
[Paper](https://arxiv.org/pdf/2412.00568)
[Github](https://github.com/PolymathicAI/the_well)

## Gray-Scott Diffusion-Reaction
Data format: Multiple files, varying number of sims, each with 5 members

### 'boundary_conditions', 2 members:

x_periodic: 1 member called 'mask', shape (128,), boolean

y_periodic: 1 member called 'mask', shape (128,), boolean

### 'dimensions', 3 members:

time: shape (1001,)

x: shape (128,)

y: shape (128,)

### 'scalars', 2 members:

F: constant, varies based on variant of GSDR

k: constant, varies based on variant of GSDR

### 't0_fields', 2 members:

A: shape (20,1001,128,128)
B: shape (20,1001,128,128)


### 't1_fields','t2_fields', 0 members:

Both empty for GSDR
