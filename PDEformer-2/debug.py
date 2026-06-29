import numpy as np
import matplotlib.pyplot as plt
from mindspore import context
from src import load_config, get_model, PDENodesCollector, sample_grf
from src.inference import infer_plot_2d, x_fenc, y_fenc, interp_fenc
import h5py
from src.data.multi_pde.pde_types import get_pde_info_cls

#LOAD MODEL
#PYNATIVE MODE for debugging, though it runs the same as GRAPH MODE, but with GRAPH MODE, you will not be able to step through/step into when debugging
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
config = load_config("configs/inference/model-M.yaml")
model = get_model(config)

# Set the data path accordingly
# raw_data_path = '/Volumes/T7/PDEFoundry2/DCR/dedalus_v5.1_DiffConvecReac2D_hom_cU1_k1e-03_0.01_seed1.hdf5'
raw_data_path = '/Volumes/T7/PDEFoundry2/MVDCR/dedalus_v5.1_MCompn2D_hom_fNL0_nv2_cU1_k1e-03_0.01_seed0.hdf5'

#Change the data class in get_pde_info_cls('XXXX') according to the dataset used
pde_type = get_pde_info_cls('mvdcr')

#Sample idx of dataset
idx_test = [0]
# idx_test = list(range(0, 30))
pde_dags_list = []

with h5py.File(raw_data_path, 'r') as f:
    for idx in idx_test:
        pde = pde_type.pde_nodes(f, idx, keep_all_coef=False)
        print(pde.node_list)
        pde_dag = pde.gen_dag(config)
        pde_dag.plot(hide='aux')
        pde_dags_list.append(pde_dag)
        print(np.array(pde_dag.node_scalar).shape)
        print(np.array(pde_dag.node_function).shape)

print(f'Total num of samples: {len(pde_dags_list)}')


#x_plot, y_plot for 128x128
#For snap_t, 4 timesteps are selected from [0,1]. However, this can be adjusted to any number of timesteps within [0,1]
x_plot, y_plot = np.meshgrid(np.linspace(0, 1, 129)[:-1], np.linspace(0, 1, 129)[:-1], indexing="ij")
snap_t=np.linspace(0, 1, 4)

# This loop is added in the event where there are more than 1 sample being used for inference
for i, dag_idx in enumerate(pde_dags_list):
    pred = infer_plot_2d(model, dag_idx, x_plot, y_plot, snap_t=snap_t)
    print('Simulation:', i)
    print(pred.shape)


