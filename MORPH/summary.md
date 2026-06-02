# Summary for MORPH
Relatively simple model to work with. Pretrained weights can be downloaded from MORPH's Github.

General pipeline: Preprocess datasets using MORPH's data_normalization scripts -> run inference using infer_MORPH.py

For reference, check the notebooks in this folder.

## Notes
Remember to change file paths in data_normalization scripts. The default data_normalization script contains all of the datasets, I have isolated some datasets used into individual data_normalization scripts. The mean and std from data normalization are saved in separate .npy files by default.

debug.py is a script used for debugging, as the name implies.

infer_MORPH.py includes plotting code by default.



