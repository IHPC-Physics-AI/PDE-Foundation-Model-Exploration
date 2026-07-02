# Results for all models

---
## DPOT (Original)
### Pretrained datasets
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

### Converted/Other datasets
Tested using default DPOT model S.

Unknown GSDR from [paper](http://arxiv.org/abs/2106.04781).

| Dataset                  | Converted Into | My result      | Paper's result (If any) | Notes                                                                 |
|--------------------------|----------------|----------------|-------------------------|-----------------------------------------------------------------------|
| TheWell GSDR             | PDEBench DR    | 0.82368        | NA                      |                                                                       |
| APEBench GSDR (gs_alpha) | PDEBench DR    | 0.64832        | NA                      |                                                                       |
| Unknown GSDR             | PDEBench DR    | 0.97508        | NA                      |                                                                       |



---
## DPOT (Pseudo-inverse)
### AFTER self.out_conv2
#### SGEDULION

| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.98898       | 0.78776                                               | 0.76104                                            | 0.75261                                             |
| Model M  | 0.97493       | 0.49768                                               | 0.47629                                            | 0.44659                                             |
| Model L  | 0.86706       | 0.54746                                               | 0.44925                                            | 0.42636                                             |

#### SGMERLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 1.02402       | 0.14069                                               | 0.11873                                            | 0.10309                                             |
| Model M  | 1.30143       | 0.13137                                               | 0.08466                                            | 0.07526                                             |
| Model L  | 1.24462       | 0.13867                                               | 0.09681                                            | 0.08274                                             |

#### SGLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.79196       | 0.32957                                               | 0.32227                                            | 0.30167                                             |
| Model M  | 0.92450       | 0.34596                                               | 0.35019                                            | 0.34973                                             |
| Model L  | 0.89457       | 0.35687                                               | 0.35631                                            | 0.35552                                             |


### BEFORE self.out_conv2
#### SGEDULION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.98898       | 0.37687                                               | 0.29432                                            | 0.28336                                             |
| Model M  | 0.97493       | 0.38375                                               | 0.29548                                            | 0.24666                                             |
| Model L  | 0.86706       | 0.37179                                               |                                                    |                                                     |


#### SGMERLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 1.02402       | 0.12046                                               | 0.06894                                            | 0.06441                                             |
| Model M  | 1.30143       | 0.09781                                               | 0.07545                                            | 0.06953                                             |
| Model L  | 0.89457       | 0.10784                                               |                                                    |                                                     |


#### SGLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.79196       | 0.25896                                               | 0.23462                                            | 0.21923                                             |
| Model M  | 0.92450       | 0.28119                                               | 0.25145                                            | 0.22434                                             |
| Model L  | 0.89457       | 0.29639                                               |                                                    |                                                     |


### Testing different reg values for BEFORE self.out_conv2
Only Model M is used here.

Best results for SGEDULION are bolded and italicized.

#### SGEDULION
| Reg   | Pseudo Inv Kernel 3, nonlinear terms 3 | Pseudo Inv Kernel 5, nonlinear terms 3 | Pseudo Inv Kernel 7, nonlinear terms 3 |
|-------|----------------------------------------|----------------------------------------|----------------------------------------|
| 1e-7  | 0.38328                                | 0.28708                                | 0.22839                                |
| 1e-6  | 0.38328                                | 0.28708                                | 0.22839                                |
| 1e-5  | 0.38328                                | **_0.28708_**                          | 0.22839                                |
| 1e-4  | 0.38328                                | 0.28709                                | 0.22839                                |
| 1e-3  | **_0.38328_**                          | 0.28713                                | 0.22838                                |
| 1e-2  | 0.38334                                | 0.28748                                | _**0.22838**_                          |
| 1e-1  | 0.38375                                | 0.28943                                | 0.22910                                |
| 1     | 0.378525                               | 0.29548                                | 0.23408                                |
| 10    | 0.38786                                | 0.30487                                | 0.24007                                |
| 100   | 0.39086                                | 0.31319                                | 0.24855                                |
| 500   | 0.39325                                | 0.31808                                | 0.26368                                |
| 1000  | 0.39465                                | 0.31934                                | 0.27162                                |
| 5000  | 0.40151                                | 0.33311                                | 0.29373                                |
| 10000 | 0.40786                                | 0.34370                                | 0.31258                                |


---
## MORPH

### Pretrained datasets
Tested using default MORPH(FM) model S.

| Dataset        | My result      | Paper's result | Notes                                       |
|----------------|----------------|----------------|---------------------------------------------|
| PDEBench DR    | 0.11177        | 0.11843        |                                             |
| PDEBench SWE   | 0.00442        | 0.00605        |                                             |
| PDEBench INS   | 0.13116        | 0.13037        | Only 4 samples used for inference           |
| -------------- | -------------- | -------------- |                                             |
| TheWell GSDR   | 0.50952        | 0.51025        | Only 'Worms' variant was used for inference |

### Converted/Other datasets
Tested using default MORPH(FM) model S.

| Dataset       | Converted Into | My result | Paper's result (If any) | Notes                                         |
|---------------|----------------|-----------|-------------------------|-----------------------------------------------|
| TheWell GSDR  | PDEBench DR    | 0.51430   | 0.51025 (TheWell GSDR)  | Only 'Bubbles' variant was used for inference |
| APEBench GSDR | PDEBench DR    | 0.57027   | NA                      | gs_alpha was used for inference               |
| APEBench GSDR | TheWell GSDR   | 0.57027   | NA                      | gs_alpha was used for inference               |

---
## PDEformer-2
### Pretrained datasets
Tested using default PDEformer-2 model M.

| Dataset          | My result | Paper's result | Notes                                |
|------------------|-----------|----------------|--------------------------------------|
| PDEFoundry DCR   | 0.098224  | 0.084          | Only 1 sample was used for inference |
| PDEFoundry MVDCR | 0.084174  | 0.135          | Only 1 sample was used for inference |

### Converted/Other datasets
Tested using default PDEformer-2 model M.

| Dataset      | Converted Into | My result | Paper's result (If any) | Notes                                         |
|--------------|----------------|-----------|-------------------------|-----------------------------------------------|
| TheWell GSDR | DAG format     | 0.419474  | NA                      | 'Bubbles' variant is used for inference       |
| PDEBench DR  | DAG format     | 3.924970  | NA                      | Only first 100 simulations used for inference |

---
## PDEtransformer
### Pretrained datasets
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


### Converted/Other datasets
Tested using default PDEtransformer(SC) model S.
Unknown GSDR from [paper](http://arxiv.org/abs/2106.04781).

| Dataset      | Converted Into | My result                                               | Paper's result (If any) | Notes                             |
|--------------|----------------|---------------------------------------------------------|-------------------------|-----------------------------------|
| Unknown GSDR | APEBench GSDR  | After 1 step: 0.693411 <br/> After 3000 steps: 0.000000 | 0.51025 (TheWell GSDR)  | Inference using gs_theta pipeline |

---
## PROSE-FD
### Pretrained datasets
Tested using prose_2to1 model.

| Dataset        | My result      | Paper's result | Notes                                                                                |
|----------------|----------------|----------------|--------------------------------------------------------------------------------------|
| PDEBench SWE   | 0.002758       | 0.0028         |                                                                                      |
| PDEBench INS   | 0.033604       | 0.0284         | Only 4 samples used for testing due to RAM limitations                               |
| -------------- | -------------- | -------------- |                                                                                      |
| PDEArena INS-U | 0.051130       | 0.0634         |                                                                                      |
| PDEArena INS-C | 0.135081       | 0.1076         |                                                                                      |
| -------------- | -------------- | -------------- |                                                                                      |
| CFDBench       | 0.002704       | 0.0054         | I only used 1 sample for each flow for testing  due to RAM limitations on my machine |

### Converted/Other datasets
Tested using prose_2to1 model.

| Dataset           | Converted Into | My result                                                | Paper's result (If any)                          | Notes                    |
|-------------------|----------------|----------------------------------------------------------|--------------------------------------------------|--------------------------|
| CFDBench Cavity   | PDEBench INS   | prop: 0.363150 <br/> geo: 0.449329 <br/> bc: 0.405691    | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench Cylinder | PDEBench INS   | prop: 0.166037   <br/> geo: 0.166919  <br/> bc: 0.163191 | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench Tube     | PDEBench INS   | prop: 0.327166 <br/> geo: 0.280711 <br/> bc: 0.202046    | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench(all)     | PDEBench INS   | 0.216937                                                 | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 | Mask values are inverted |




