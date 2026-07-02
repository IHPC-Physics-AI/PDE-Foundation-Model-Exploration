# Results

## Pretrained datasets
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

## Converted/Other datasets
Tested using prose_2to1 model.

| Dataset           | Converted Into | My result                                                | Paper's result (If any)                          | Notes                    |
|-------------------|----------------|----------------------------------------------------------|--------------------------------------------------|--------------------------|
| CFDBench Cavity   | PDEBench INS   | prop: 0.363150 <br/> geo: 0.449329 <br/> bc: 0.405691    | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench Cylinder | PDEBench INS   | prop: 0.166037   <br/> geo: 0.166919  <br/> bc: 0.163191 | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench Tube     | PDEBench INS   | prop: 0.327166 <br/> geo: 0.280711 <br/> bc: 0.202046    | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 |                          |
| CFDBench(all)     | PDEBench INS   | 0.216937                                                 | CFDBench(all): 0.0054 <br/> PDEBench INS: 0.0284 | Mask values are inverted |







