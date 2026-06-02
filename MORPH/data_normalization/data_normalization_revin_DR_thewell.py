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
loadpath_dr2d  = cfg['DR2d_data_pdebench']['file_path_dr2d']

# create folders if they don't exist
for base in (loadpath_dr2d,):
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
# savepath_norm_data_dr2d = cfg['DR2d_data_pdebench']['file_path_dr2d_n']
savepath_norm_data_dr2d = "/Volumes/T7/MORPH_Processed/normalized_revin/DR2d_data_pdebench"

for base in (savepath_norm_data_dr2d,):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed

# create folders if they don't exist
for base in (savepath_norm_data_dr2d,):
    for split in ('train','valid','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)


#%% DR2D data
from src.utils.dataloaders.dataloader_dr import split_and_save_h5, DR2DDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_dr2d,
                  savepath = loadpath_dr2d,
                  dataset_name='DR',
                  train_frac = 0.0,
                  rand = True)

# load the splited raw DR data
loader = DR2DDataLoader(loadpath_dr2d)
# train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
# dataset = np.concatenate((train,val,test), axis = 0)
dataset = test
print("Shape of inflated DR data", dataset.shape)     # (N, T, D, H, W, C, F)

# Reshape & expand dims for RevIN (N, T, D, H, W, C, F) -> (N, T, F, C, D, H, W)
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
print("Transposed DR data", dataset_tr.shape)

# compute & normalize
rev_dr.compute_stats(dataset_tr, prefix='stats_dr')
dataset_dr_norm = rev_dr.normalize(dataset_tr, prefix='stats_dr')
print("Normalize dataset shape", dataset_dr_norm.shape)

# Check for DR dataset
# Check round‐trip via denormalize
tol_2 = 1e-5
recovered = rev_dr.denormalize(dataset_dr_norm, prefix='stats_dr')
print("Denormalized dataset shape", recovered.shape)

max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_2, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_dr_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)[:,:,:,:,0,:] # since C is repeated, taking only one
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_dr2d) if f.endswith('.h5') or f.endswith('.hdf5')]
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_dr2d, filename)
with h5py.File(out_path, 'w') as f_out:
    for i in range(dataset_sq.shape[0]): # since test and val are repeated
        grp = f_out.create_group(f"{i:04d}")
        grp.create_dataset('data', data = dataset_sq[i], compression='lzf')
print("Saved normalized DR to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_norm_data_dr2d, savepath_norm_data_dr2d,
                  dataset_name = 'DR',
                  train_frac = 0.0,
                  rand = False)

