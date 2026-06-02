import os
import numpy as np
import h5py
import sys

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# load the classes
from src.utils.normalization import RevIN
from config.data_config_vis import DataConfig

# raw data directory
dataset_dir = "/Volumes/T7/MORPH_Processed"

# instantiate the class
cfg = DataConfig(dataset_dir=dataset_dir, project_root=project_root)

# load paths
load_root = "/Volumes/T7/MORPH_Processed"

# load all the filepaths
# --- Pretraining ---
loadpath_sw2d  = cfg['2dSW_pdebench']['file_path_sw2d']
# loadpath_sw2d = "/Volumes/T7/PDEBench/2D/shallow-water"

# create folders if they don't exist
for base in (loadpath_sw2d,):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

# savepath of mu and var
savepath_muvar = os.path.join(project_root, 'data')

# reversible instance normalization
(rev_sw2d) = (RevIN(savepath_muvar))

# savepath of normalized data
# --- Pretraining ---
# savepath_norm_data_sw2d = cfg['2dSW_pdebench']['file_path_sw2d_n']
savepath_norm_data_sw2d = "/Volumes/T7/MORPH_Processed/normalized_revin/2dSW_pdebench"

# ensure the parent trees exist:
# - datasets/normalized_revin
# - datasets/normalized_revin/MHD3d_data
# - datasets/normalized_revin/DR_data
for base in (savepath_norm_data_sw2d,):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed

# create folders if they don't exist
for base in (savepath_norm_data_sw2d,):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

#%% --->> PRETRAINING DATASETS
#%% SW2D data
'''
data loading and processing similar to DR dataset.
'''
from src.utils.dataloaders.dataloader_sw2d import split_and_save_h5, SW2dDataLoader

print(f"Checking path: {os.path.abspath(loadpath_sw2d)}")
print(f"Files found: {os.listdir(loadpath_sw2d)}")

# first split the raw SW data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_sw2d,
                  savepath = loadpath_sw2d,
                  dataset_name='SW2d',
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = SW2dDataLoader(loadpath_sw2d)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated SW data", dataset.shape)

# Reshape & expand dims for RevIN
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed SW data", dataset_tr.shape)

# compute & normalize
rev_sw2d.compute_stats(dataset_tr, prefix='stats_sw')
dataset_sw2d_norm = rev_sw2d.normalize(dataset_tr, prefix='stats_sw')
print("Normalize dataset shape", dataset_sw2d_norm.shape)

# Check for SW2d dataset
# Check round‐trip via denormalize
tol_4 = 1e-5
recovered = rev_sw2d.denormalize(dataset_sw2d_norm, prefix='stats_sw')
max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_4, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_sw2d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)[:,:,:,:,0,:] # squeeze C dim (if C is repeated, use 1)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_sw2d) if f.endswith('.h5') or f.endswith('.hdf5')]
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_sw2d, filename)
with h5py.File(out_path, 'w') as f_out:
    for i in range(dataset_sq.shape[0]): # since test and val are repeated
        grp = f_out.create_group(f"{i:04d}")
        grp.create_dataset('data', data = dataset_sq[i], compression='lzf')
print("Saved normalized SW to", out_path)

# split the normed SW data into train/test/val
split_and_save_h5(savepath_norm_data_sw2d, savepath_norm_data_sw2d,
                  dataset_name = 'SW2d',
                  train_frac = 0.8,
                  rand = False)
