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
loadpath_cfd2dic  = cfg['2dcfd_ic_pdebench']['file_path_cfd2d_ic']

# create folders if they don't exist
for base in (loadpath_cfd2dic,):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

# savepath of mu and var
savepath_muvar = os.path.join(project_root, 'data')

# reversible instance normalization
(rev_mhd, rev_dr, rev_cfd1d, rev_sw2d, rev_cfd2dic, rev_cfd3d,
 rev_dr1d, rev_cfd2d, rev_be1d, rev_cfd3d_turb, rev_gsdr2d, rev_tgc3d, rev_fns_kf_2d,
 rev_ce_2d, rev_ns_2d) = (RevIN(savepath_muvar) for _ in range(15))

# savepath of normalized data
# --- Pretraining ---
savepath_norm_data_cfd2dic = '/Volumes/T7/MORPH_Processed/normalized_revin/2dcfd_ic_pdebench'

# ensure the parent trees exist:
# - datasets/normalized_revin
# - datasets/normalized_revin/MHD3d_data
# - datasets/normalized_revin/DR_data
for base in (savepath_norm_data_cfd2dic,):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed

# create folders if they don't exist
for base in (savepath_norm_data_cfd2dic,):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)


#%% CFD2d (IC)
####################### Load and process CFD2d-IC data ##############################
from src.utils.dataloaders.dataloader_cfd2dic import split_and_save_h5, CFD2dicDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_dir = loadpath_cfd2dic,
                  out_dir = loadpath_cfd2dic,
                  dataset_name='cfd2dic',
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = CFD2dicDataLoader(loadpath_cfd2dic, force = True)
# train, val = loader.split_train()          # data is already inflated
# test = loader.split_test()
# dataset = np.concatenate((train,val,test), axis = 0)
dataset = loader.split_test()
# del train, val, test
print("Shape of inflated CFD2d data", dataset.shape)

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD2d data", dataset.shape)

# compute & normalize
rev_cfd2dic.compute_stats(dataset, prefix='stats_cfd2d-ic')
dataset_cfd2dic_norm = rev_cfd2dic.normalize(dataset, prefix='stats_cfd2d-ic')
print("Normalize dataset shape", dataset_cfd2dic_norm.shape)

# Check round‐trip via denormalize
tol_5 = 1e-5
recovered = rev_cfd2dic.denormalize(dataset_cfd2dic_norm, prefix='stats_cfd2d-ic')
max_error = 0.0
for i in range(recovered.shape[0]):
    # print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_5, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_cfd2dic_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2) # Final shape (N,T,H,W,C,F)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
# load raw files
raw_files = sorted([f for f in os.listdir(loadpath_cfd2dic)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
# splits= 4
N = dataset_sq.shape[0]
# chunk_size = N // splits
splits = len(raw_files)
chunk_size = max(1, N // splits)
# Loop over each chunk and save
for i, fname in enumerate(raw_files):
    print(f'Processing: {raw_files}...')
    start = i * chunk_size
    end   = (i + 1) * chunk_size
    chunk = dataset_sq[start:end]  # shape (chunk_size, T, H, W, C, F)

    # reconstruct force & velocity in raw form
    # force: pick either the first timestep (all fx, fy are the same over T)
    force_chunk = chunk[:, 0, :, :, :, 0]    # → (m, H, W, 2)
    # velocity: take the full time‐series
    vel_chunk   = chunk[..., 1]   # → (m, T, H, W, 2)

    out_path = os.path.join(savepath_norm_data_cfd2dic, fname)
    with h5py.File(out_path, "w") as f5:
        # assuming F == 3 and channel order [Vx, density, pressure]
        f5.create_dataset("force",    data=force_chunk, compression="gzip")
        f5.create_dataset("velocity",  data=vel_chunk, compression="gzip")

    print(f"Saved force: {force_chunk.shape}, vel: {vel_chunk.shape} trajectories to {fname}")

# split the normed CFD data into train/test/val
split_and_save_h5(savepath_norm_data_cfd2dic, savepath_norm_data_cfd2dic,
                  dataset_name = 'cfd2dic',
                  train_frac = 0.8,
                  rand = False)
