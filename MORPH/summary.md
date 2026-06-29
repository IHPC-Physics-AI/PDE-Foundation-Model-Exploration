# Summary for MORPH
Relatively simple model to work with. Pretrained weights can be downloaded from MORPH's Github.

General pipeline: Preprocess datasets using MORPH's data_normalization scripts -> run inference using infer_MORPH.py

For reference, check the notebooks in this folder.

## Notes
Remember to change file paths in data_normalization scripts.

## data_normalization
The default data_normalization script contains all of the datasets, I have isolated some datasets used into individual data_normalization scripts for easier usage. 
The mean and std from data normalization are saved in separate .npy files by default.

The only change between each data_normalization script are the data paths.

For example, apeDRalpha_to_PDBdr's script is the same as DR's script, just that the data path point towards APEBench gs_alpha instead of DR.

## infer_morph.py
No need to tweak/add anything. The script saves the results and the plots under experiments/results.

## debug.py
Primarily used for debugging and inference testing (using infer_MORPH.py)

Arguments used are based on MORPH's paper. 

```python
    sys.argv = [
        "infer_MORPH.py",
        "--model_choice", "FM",
        "--model_size", "S",
        "--checkpoint", "morph-S-FM-max_ar1_ep225.pth",
        "--test_dataset", "GSDR2D",
        "--ar_order", "1",
        "--rollout_horizon", "10",
        "--batch_size", "1",
        "--test_sample", "0",
        "--max_ar_order", "1"
]
```

Note that model_choice should be FM, unless you are testing something else. The only weights given are the weights for FM.

ar_order/rollout_horizon/max_ar_order values are based on MORPH's paper, as mentioned.



