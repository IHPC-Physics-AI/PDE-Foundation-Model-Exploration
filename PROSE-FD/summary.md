# Summary for PROSE-FD

Pretrained weights can be downloaded from PROSE-FD's Github

Remember to change data paths in configs/data/fluids.yaml AND configs/data/fluids_sample.yaml, and in convert_cfdbench.py


## Edit np.infty to np.inf in trainer.py (line 63)
```python
# Original
self.best_metrics = {metric: (-np.infty if biggest else np.infty) for (metric, biggest) in self.metrics}

# Edited
self.best_metrics = {metric: (-np.inf if biggest else np.inf) for (metric, biggest) in self.metrics}
```

## Preprocessing
PDEBench/PDEArena datasets do not require any preprocessing before inference.

CFDBench requires preprocessing, and the scripts used are taken from DPOT's own CFDBench data preprocessing. 

So, if you have already preprocessed CFDBench dataset when running DPOT, you can use them for PROSE-FD as well.

I have added some print statements in cylinder.py/dam.py/tube.py for checks during preprocessing.

For example in cylinder.py:

```python
def get_cylinder_datasets:
    ...
    print("==== Number of cases in different splits ====")
    print(f"train: {len(train_case_dirs)}, " f"dev: {len(dev_case_dirs)}, " f"test: {len(test_case_dirs)}")
    print("=============================================")
    ...

def get_cylinder_autodatasets:
    ...
    print("\n--- DATA SPLIT MAPPING ---")
    print(f"TRAIN CASES: {[d.name for d in train_case_dirs]}")
    print(f"DEV CASES:   {[d.name for d in dev_case_dirs]}")
    print(f"TEST CASES:  {[d.name for d in test_case_dirs]}")
    print("--------------------------\n")


    print("==== Number of cases in different splits ====")
    print(f"train: {len(train_case_dirs)}, " f"dev: {len(dev_case_dirs)}, " f"test: {len(test_case_dirs)}")
    print("=============================================")
    ...
```

## Inference and OOM
If your machine does not have enough RAM and Google Colab does not assign you any GPU runtime, you can edit the number of samples used for CFDBench preprocessing like so:

```python
    # Original
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    
    # Edited
    num_train = 1
    num_dev = 1
    num_test = 1
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev : num_train + num_dev + num_test]
```

As per the code block above, we are now only testing a singular sample, which your machine should have enough RAM for. 

Note that the edits should be made for each of the respective FLOW_TYPE.py files, under get_FLOWTYPE_datasets and get_FLOWTYPE_autodatasets.



## Conversion Notebooks
I have attempted to convert CFDBench flows to PDEBench INS dataset format. The methods are explained in greater detail in the respective notebooks under Notebooks_otherData.

However, after some attempts and discussion with Dr Ooi Chin Chun and Dr Wei Zhao, we have concluded that CFDBench flows/PDEBench INS/PDEArena INS datasets should be considered as unique problems, instead of grouping them together under INS.

## evaluate.py
I have added some code blocks in evaluate.py to check the shape of certain variables at inference.

Lines 126-155
```python
for type, loader in self.dataloaders.items():

    #FOR CHECKS
    print('Type:', type)
    #
    
    ...

    for idx, samples in enumerate(loader):
        bs = len(samples["data"])
        eval_size += bs
        model_input, d = self.trainer.prepare_data(samples, train=False)

        #FOR CHECKS
        print('Model input:', model_input.keys())
        print(model_input['input_times'].shape)
        print(model_input['output_times'].shape)
        print(model_input['data_input'].shape)
        print(model_input['symbol_input'])
        print(model_input['symbol_padding_mask'])

        print('d:', d.keys())
        print(d['data_label'].shape)
        print(d['data_mask'].shape)
        # print(d['mean'])
        # print(d['std'])
        ##
```



