# Summary for PDEtransformer
Focusing on separate channels version. Mixed channels is the simpler version.

Pretrained weights can be downloaded from PDEtransformer's Github.

General pipeline: Load dataset and model -> Inference

For the best reference, check the notebooks in PDEtransformer's Github. I have included my own notebooks, which are used for testing pretrained/other data.

## Notes
Remember to change data paths in dataset.py and configs.

## pde_transformer.py
The paper suggests that there is a conditioning vector that holds metadata (i.e. feed rate/kill rate, etc). 

Tracing the input, there is indeed metadata being passed along, but in pde_transformer.py, PDE_PARAMETER_SCALING = 0, which is then multiplied with the encoded pde parameters, which essentially voids their input into the model.

pde_transformer.py/PDEImpl:

```python
    # Lines 1285 to 1291
    assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
    self.num_encoder_layers = len(depth) // 2
    
    self.learn_sigma = learn_sigma
    self.in_channels = num_timesteps
    self.out_channels = num_timesteps
    
    self.SIMULATION_TIME_SCALING = 0.0
    self.SIMULATION_DT_SCALING = 0.0
    self.PDE_PARAMETER_SCALING = 0.0 #<----- Here

    # AND
    
    # Within forward pass
    # Lines 1498 to 1499
    c = (... + self.PDE_PARAMETER_SCALING * pde_parameter_emb)
```

## debug.py
Primarily used for debugging and inference testing.

For Macbook users, set num_workers = 0

unrolling_steps: The number of timesteps being predicted. I set this to 1 since I only want single step predictions.

test_unrolling_steps: Total number of timesteps used for testing. Since the GSDR datasets used from APEBench have 30 timesteps, I set this value to 29 (29 predictions + 1 input frame).

For GSDR, normalize_data is set to 'mean-std', and normalize_const can be ignored, since it is hardcoded to be None for all GSDR variants, regardless of the value set.


