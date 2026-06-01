# PDEBench Datasets
Only focusing on DR/SW/INS. Download instructions can be found in PDEBench's Github
[Paper](https://arxiv.org/pdf/2210.07182)
[Github](https://github.com/pdebench/PDEBench)

## Diffusion-Reaction
Data format: Singular file, 1000 sims, each with 2 members ('data', 'grid')

'data': shape (101, 128, 128, 2)

'grid', 3 members:  't' (101,), 'x' (128,), 'y' (128,)

## Shallow water
Data format: Singular file, 1000 sims, each with 2 members ('data', 'grid')

'data': shape (101, 128, 128, 1)

'grid', 3 members:  't' (101,), 'x' (128,), 'y' (128,)

## Incompressible Navier-Stokes
Data format: Multiple files, each with 4 sims.

Each file has 4 members: 'force', 'particles', 't', 'velocity'

'force': shape (4, 512, 512, 2)

'particles': shape (4, 1000, 512, 512, 1)

't': shape (4, 1000)

'velocity': shape (4, 1000, 512, 512, 2)

