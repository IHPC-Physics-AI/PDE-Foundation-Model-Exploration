# Results

## Pretrained datasets
Tested using default MORPH(FM) model S.

| Dataset        | My result      | Paper's result | Notes                                       |
|----------------|----------------|----------------|---------------------------------------------|
| PDEBench DR    | 0.11177        | 0.11843        |                                             |
| PDEBench SWE   | 0.00442        | 0.00605        |                                             |
| PDEBench INS   | 0.13116        | 0.13037        | Only 4 samples used for inference           |
| -------------- | -------------- | -------------- |                                             |
| TheWell GSDR   | 0.50952        | 0.51025        | Only 'Worms' variant was used for inference |

## Converted/Other datasets
Tested using default MORPH(FM) model S.

| Dataset       | Converted Into | My result | Paper's result (If any) | Notes                                         |
|---------------|----------------|-----------|-------------------------|-----------------------------------------------|
| TheWell GSDR  | PDEBench DR    | 0.51430   | 0.51025 (TheWell GSDR)  | Only 'Bubbles' variant was used for inference |
| APEBench GSDR | PDEBench DR    | 0.57027   | NA                      | gs_alpha was used for inference               |
| APEBench GSDR | TheWell GSDR   | 0.57027   | NA                      | gs_alpha was used for inference               |






