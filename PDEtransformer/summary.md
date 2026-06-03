# Summary for PDEtransformer
Focusing on separate channels version. Mixed channels is the simpler version.

Pretrained weights can be downloaded from PDEtransformer's Github.

General pipeline: 

For the best reference, check the notebooks in PDEtransformer's Github. I have included my own notebooks, which are used for testing pretrained/other data.

## Notes
The paper suggests that there is a conditioning vector that holds metadata (i.e. feed rate/kill rate, etc). This is only true to a certain extent. Tracing the input, there is indeed metadata being passed along, but in pde_transformer.py, PDE_PARAMETER_SCALING == 0, which is then multiplied with the encoded pde parameters, which essentially voids their input into the model.

In summary, the model seems to be purely data-based, even though pde parameters are used as input into the model (see tasks.py and pde_transformer.py). For clarity, I would recommend tracing the input.

Remember to change data paths in dataset.py and configs.

debug.py for debugging, as the name implies.

## 
