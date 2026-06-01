# PDEFoundry Datasets
Only focusing on DCR/MVDCR. Download instructions can be found in PDEFoundry's Github
[Paper](https://arxiv.org/pdf/2507.15409)
[Github](https://github.com/functoreality/pdefoundry-2/tree/main)

## Diffusion-Convection-Reaction
There are multiple variants of DCR that can be downloaded, but this section will only elaborate on the base variant.

The other variants are similar in terms of structure.

Data format: 5 members for each .h5 file

### 'args', 11 members:
['coef_magnitude', 'inhom_diff_u', 'kappa_max', 'kappa_min', 'np_seed', 'num_pde', 'num_sinusoid', 'num_sol_per_npy', 'plot_results', 'print_coef_level', 'u_bound']

### 'coef', 6 members:
* **`Lu`** (2 members)
    * Contains `diff_type` `(1000,)` and `value` `(1000,)`
* **`f0`**: Shape `(1000, 1, 4)`
* **`f1`**: Shape `(1000, 1, 4)`
* **`f2`**: Shape `(1000, 1, 4)`
* **`s`** (2 members)
    * Contains `coef_type` `(1000,)` and `field` `(1000, 128, 128)`
* **`u_ic`**: Shape `(1000, 128, 128)`

### 'coord', 3 members:
t: (101,)

x: (128, 1)

y: (1, 128)

### 'pde_info', 4 members:
['is_inverse', 'pde_type_id', 'preprocess_dag', 'version']

### 'sol', 1 member:
'u', shape (1000, 101, 128, 128)


## Multi-Variable Diffusion-Convection-Reaction
There are multiple variants of MVDCR that can be downloaded, but this section will only elaborate on the base variant.

The other variants are similar in terms of structure.

Data format: 5 members for each .h5 file

### 'args', 11 members:
['coef_magnitude', 'flux_num_nonlin', 'inhom_diff_u', 'kappa_max', 'kappa_min', 'n_vars', 'np_seed', 'num_pde', 'num_sol_buffer', 'num_sol_per_npy', 'periodic', 'plot_results', 'print_coef_level', 'record_failed_samples', 'resume', 'terminate_on_save', 'u_bound']

### 'coef', 6 members:
* **`Lu`** (2 groups)
    * **`0`**: Contains `diff_type` `(1000,)` and `value` `(1000,)`
    * **`1`**: Contains `diff_type` `(1000,)` and `value` `(1000,)`
* **`f0`** (2 groups)
    * **`deg2`**: Contains `coo_i`, `coo_j`, `coo_k`, `coo_len`, `coo_vals` (All shape `(1000, 4)`, except coo_len `(1000,)`)
    * **`lin`**: Contains `coo_i`, `coo_j`, `coo_len`, `coo_vals` (All shape `(1000, 4)`, except coo_len `(1000,)`)
* **`f1`, `f2`** (2 groups)
    * **`deg2`**: Contains `coo_i`, `coo_j`, `coo_k`, `coo_len`, `coo_vals` (All shape `(1000, 0)`, except coo_len `(1000,)`)
    * **`lin`**: Contains `coo_i`, `coo_j`, `coo_len`, `coo_vals` (All shape `(1000, 4)`, except coo_len `(1000,)`)
* **`s`** * Contains `coef_type` `(1000,)` and `field` `(1000, 128, 128)`
* **`u_ic`** (2 groups)
    * **`0`**: Shape (1000,128,128)
    * **`1`**: Shape (1000,128,128)

### 'coord', 3 members:
t: (101,)

x: (128, 1)

y: (1, 128)

### 'pde_info', 4 members:
['is_inverse', 'pde_type_id', 'preprocess_dag', 'version']

### 'sol', 2 members:
* **`sol`** (2 groups)
    * **`u0`**: Shape (1000, 101, 128, 128)
    * **`u1`**: Shape (1000, 101, 128, 128)
