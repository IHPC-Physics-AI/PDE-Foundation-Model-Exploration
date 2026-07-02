# Results

## Pretrained datasets
Tested using default PDEtransformer(MC) model S.

| Dataset             | My result                                             | Paper's result                                    | Notes |
|---------------------|-------------------------------------------------------|---------------------------------------------------|-------|
| APEBench gs_alpha   | After 1 step: 0.024236 <br/> After 20 steps: 0.608092 | After 1 step: 0.0248 <br/> After 20 steps: 0.6696 |       |
| APEBench gs_beta    | After 1 step: 0.027490 <br/> After 20 steps: 0.613758 | After 1 step: 0.0295 <br/> After 20 steps: 0.6643 |       |
| APEBench gs_gamma   | After 1 step: 0.033284 <br/> After 20 steps: 0.785972 | After 1 step: 0.0316 <br/> After 20 steps: 0.7887 |       |
| APEBench gs_epsilon | After 1 step: 0.019647 <br/> After 20 steps: 0.325716 | After 1 step: 0.0175 <br/> After 20 steps: 0.3196 |       |
| APEBench gs_theta   | After 1 step: 0.011252 <br/> After 20 steps: 0.987747 | After 1 step: 0.0129 <br/> After 20 steps: 1.0247 |       |
| APEBench gs_iota    | After 1 step: 0.016844 <br/> After 20 steps: 0.708248 | After 1 step: 0.0141 <br/> After 20 steps: 0.7601 |       |
| APEBench gs_kappa   | After 1 step: 0.026268 <br/> After 20 steps: 0.665534 | After 1 step: 0.0231 <br/> After 20 steps: 0.7272 |       |


## Converted/Other datasets
Tested using default PDEtransformer(SC) model S.

| Dataset      | Converted Into | My result                                               | Paper's result (If any) | Notes                             |
|--------------|----------------|---------------------------------------------------------|-------------------------|-----------------------------------|
| Unknown GSDR | APEBench GSDR  | After 1 step: 0.693411 <br/> After 3000 steps: 0.000000 | 0.51025 (TheWell GSDR)  | Inference using gs_theta pipeline |

Unknown GSDR from [paper](http://arxiv.org/abs/2106.04781).







