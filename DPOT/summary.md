# Summary for DPOT
Relatively simple model to work with. Pretrained weights can be downloaded from DPOT's Github.

General pipeline: Preprocess data using DPOT's script -> Run inference using DPOT's script (evaluate.py)

For reference, check the notebooks in this folder.

## Notes
Remember to change file paths in make_master_file.py.

evaluate.py scripts include custom plotting code, which were not included in DPOT's original evaluate.py script. The animation/plotting blocks are indicated by comments.

For Macbook users, it is important to set num_workers=0 for the train/test loaders in evaluate.py as seen below:

```python
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
    test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0) for test_dataset in test_datasets]
```

## debug.py
Primary use of this script is to debug the model, though it can be used for inference testing as well.

For Model S, use debug.py as it is.

For Model M, arguments to add are:
```python
    "--n_layers", "12", 
    "--mlp_ratio", "4", 
```

For Model L, arguments to add are:
```python
    "--out_layer_dim", "128", 
    "--width", "1536", 
    "--n_layers", "24", 
    "--data_weights", "8", "2", "8", "1", "1", "1", "1", "15", "20", "1", "3", "1", 
    "--n_blocks", "16", 
    "--mlp_ratio", "4", 
```

The respective arguments above are taken from either the configs or the default/commented out arguments in evaluate.py.

## evaluate.py VS evaluate_unknown.py
Dataset used: PDEBench 2D Diffusion-Reaction

evaluate.py: Works autoregressively, i.e. takes t0-9 as input to predict t10, before sliding 1 frame forward, taking t1-10 to predict t11, till t100. It is unable to go beyond this timestep, since PDB DR only has 101 timesteps.

evaluate_unknown.py: 

The goal of this script is to predict timesteps 101-110, i.e. the next 10 timesteps with no ground truth/reference.

From evaluate_unknown.py:

```python
            N_EXTRAP = 10
            extrap_preds = [] 
            extrap_xx = yy[..., -args.T_in:, :] # Use ground truth t=91-100 (real)
            print(f'extrap_xx seeded with real GT: {extrap_xx.shape}')

            #Feed t91-100 into model for autoregressive prediction
            for step in range(N_EXTRAP):
                im_ext, _ = model(extrap_xx)
                extrap_preds.append(im_ext)
                extrap_xx = torch.cat((extrap_xx[..., args.T_bundle:, :], im_ext), dim=-2)
                print(f'  [Extrap] t={101 + step} | extrap_xx: {extrap_xx.shape}')
            extrap_pred = torch.cat(extrap_preds, dim=-2)
            extrap_np = extrap_pred.squeeze(0).cpu().numpy()  # (128, 128, N_EXTRAP, 4)
```

We take the last 10 time steps from yy (ground truth), before autoregressively predicting t101-110 in the following loop.

The same sliding window used for evaluate.py is also used in the loop here.

## DPOT Forward pass
The image below illustrates the shape changes of the input for a single forward pass. For 1 sample, there are 91 forward passes, since there are 91 predictions, and 10 initial input frames.

![1 Forward pass](./dpot_forwardpass.drawio.png)