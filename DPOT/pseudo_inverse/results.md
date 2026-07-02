# Results for SGEDULION/SGMERLION/SGLION using DPOT Pseudo-inverse model

Inference done using Google Colab T4 (Free tier).

## SGEDULION/SGMERLION/SGLION notebooks

Every single cell in the notebooks has the exact same code, which is used to plot the pred .h5 file that was saved after inference.

Only difference between each of the blocks are the data paths. 

The plot is essentially an animation over all 91 predicted steps, in the format:
```
| Original DPOT Pred  | Ground Truth | Absolute Error |

| Pseudo Inverse Pred | Ground Truth | Absolute Error |
```



## AFTER self.out_conv2

```python
        # pseudo-inverse

        x = self.out_deconv(x) #S/M: (1,1024,16,16), L: (1,1536,16,16)

        x = self.out_act1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_conv1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_act2(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_conv2(x) #S/M: (1,32,128,128) -> (1,4,128,128), L: (1,128,128,128) -> (1,4,128,128)
        
        x, ssr = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(x, inputs0)   # <--- Here
```

### SGEDULION

| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.98898       | 0.78776                                               | 0.76104                                            | 0.75261                                             |
| Model M  | 0.97493       | 0.49768                                               | 0.47629                                            | 0.44659                                             |
| Model L  | 0.86706       | 0.54746                                               | 0.44925                                            | 0.42636                                             |

### SGMERLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 1.02402       | 0.14069                                               | 0.11873                                            | 0.10309                                             |
| Model M  | 1.30143       | 0.13137                                               | 0.08466                                            | 0.07526                                             |
| Model L  | 1.24462       | 0.13867                                               | 0.09681                                            | 0.08274                                             |

### SGLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.79196       | 0.32957                                               | 0.32227                                            | 0.30167                                             |
| Model M  | 0.92450       | 0.34596                                               | 0.35019                                            | 0.34973                                             |
| Model L  | 0.89457       | 0.35687                                               | 0.35631                                            | 0.35552                                             |


## BEFORE self.out_conv2
Remember to comment out all layers after the pseudo-inverse function.

No results for Model L kernel 5/7, due to GPU OOM.

```python
        # pseudo-inverse

        x = self.out_deconv(x) #S/M: (1,1024,16,16), L: (1,1536,16,16)

        x = self.out_act1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_conv1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_act2(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x, ssr = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(x, inputs0)   # <--- Here

        # x = self.out_conv2(x) #S/M: (1,32,128,128) -> (1,4,128,128), L: (1,128,128,128) -> (1,4,128,128) # <--- Comment this out
```

### SGEDULION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.98898       | 0.37687                                               | 0.29432                                            | 0.28336                                             |
| Model M  | 0.97493       | 0.38375                                               | 0.29548                                            | 0.24666                                             |
| Model L  | 0.86706       | 0.37179                                               |                                                    |                                                     |


### SGMERLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 1.02402       | 0.12046                                               | 0.06894                                            | 0.06441                                             |
| Model M  | 1.30143       | 0.09781                                               | 0.07545                                            | 0.06953                                             |
| Model L  | 0.89457       | 0.10784                                               |                                                    |                                                     |


### SGLION
| -------- | Original DPOT | Pseudo Inv Kernel 3 <br/> Reg 1e-1, nonlinear terms 3 | Pseudo Inv Kernel 5 <br/> Reg 1, nonlinear terms 3 | Pseudo Inv Kernel 7 <br/> Reg 75, nonlinear terms 3 |
|----------|---------------|-------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| Model S  | 0.79196       | 0.25896                                               | 0.23462                                            | 0.21923                                             |
| Model M  | 0.92450       | 0.28119                                               | 0.25145                                            | 0.22434                                             |
| Model L  | 0.89457       | 0.29639                                               |                                                    |                                                     |


# Testing different reg values for BEFORE self.out_conv2
Only Model M is used here.

Best results for SGEDULION are bolded and italicized.

### SGEDULION
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