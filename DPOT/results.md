# Results

## Pretrained datasets
Tested using default DPOT model S.

| Dataset        | My result      | Paper's result | Notes                                                                               |
|----------------|----------------|----------------|-------------------------------------------------------------------------------------|
| PDEBench DR    | 0.03787        | 0.0379         |                                                                                     |
| PDEBench SWE   | 0.00623        | 0.00657        |                                                                                     |
| -------------- | -------------- | -------------- |                                                                                     |
| PDEArena INS-U | 0.09984        | 0.0991         |                                                                                     |
| PDEArena INS-C | 0.31226        | 0.316          |                                                                                     |
| -------------- | -------------- | -------------- |                                                                                     |
| CFDBench       | 0.00461        | 0.00696        | I only used 1 sample for each flow for testing due to RAM limitations on my machine |

## Converted/Other datasets
Tested using default DPOT model S.

| Dataset                  | Converted Into | My result      | Paper's result (If any) | Notes                                                                 |
|--------------------------|----------------|----------------|-------------------------|-----------------------------------------------------------------------|
| TheWell GSDR             | PDEBench DR    | 0.82368        | NA                      |                                                                       |
| APEBench GSDR (gs_alpha) | PDEBench DR    | 0.64832        | NA                      |                                                                       |
| Unknown GSDR             | PDEBench DR    | 0.97508        | NA                      |                                                                       |

Unknown GSDR from [paper](http://arxiv.org/abs/2106.04781).

## Pseudo-inverse DPOT
Check results under pseudo_inverse/results.md






