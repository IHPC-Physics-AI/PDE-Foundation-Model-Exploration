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
# --- Finetuning ---
loadpath_gsdr2d = cfg['2dgrayscottdr_thewell']['file_path_2dgsdr']

# create folders if they don't exist
for base in (loadpath_gsdr2d,):
    # for split in ('train','val','test'):
    #     os.makedirs(os.path.join(base, split), exist_ok=True)
    for split in ['test']:
        os.makedirs(os.path.join(base, split), exist_ok=True)

# savepath of mu and var
savepath_muvar = os.path.join(project_root, 'data')

# reversible instance normalization
(rev_mhd, rev_dr, rev_cfd1d, rev_sw2d, rev_cfd2dic, rev_cfd3d,
 rev_dr1d, rev_cfd2d, rev_be1d, rev_cfd3d_turb, rev_gsdr2d, rev_tgc3d, rev_fns_kf_2d,
 rev_ce_2d, rev_ns_2d) = (RevIN(savepath_muvar) for _ in range(15))

# savepath of normalized data
# --- Finetuning ---
# savepath_norm_data_gsdr2d = cfg['2dgrayscottdr_thewell']['file_path_2dgsdr_n']
savepath_norm_data_gsdr2d = "/Volumes/T7/MORPH_Processed/normalized_revin/2dgrayscottdr_thewell"

# ensure the parent trees exist:
# - datasets/normalized_revin
# - datasets/normalized_revin/MHD3d_data
# - datasets/normalized_revin/DR_data
for base in (savepath_norm_data_gsdr2d,):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed

# create folders if they don't exist
for base in (savepath_norm_data_gsdr2d,):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)


#%% GSDR-2D
from src.utils.dataloaders.dataloader_gsdr2d import GSDR2dDataLoader

dataset_gsdr = GSDR2dDataLoader(loadpath_gsdr2d)
# train_data, val_data = dataset_gsdr.split_train()
test_data = dataset_gsdr.split_test()
# dataset_gsdr = np.concatenate((train_data,val_data,test_data), axis = 0)
dataset_gsdr = test_data
print("Shape of GSDR2D data", dataset_gsdr.shape)

# Reshape GSDR data into (N,T,F,C,D,H,W)
dataset_gsdr = dataset_gsdr.transpose(0, 1, 6, 5, 2, 3, 4)
print("Reshape of MHD data", dataset_gsdr.shape)

# calculate revin stats for MHD data and store it
rev_gsdr2d.compute_stats(dataset_gsdr, prefix='stats_gsdr2d')

# normalize the data
dataset_gsdr_norm = rev_gsdr2d.normalize(dataset_gsdr, prefix='stats_gsdr2d')
print("Normalize dataset shape", dataset_gsdr_norm.shape)

# Checks for GSDR ReVIN
tol_1 = 1e-4
# Check round‐trip via denormalize
recovered = rev_gsdr2d.denormalize(dataset_gsdr_norm, prefix='stats_gsdr2d')
diff = np.abs(recovered - dataset_gsdr)
print(f"Round-trip max abs error: {diff.max():.3e}")
assert diff.max() < tol_1, "Denormalization did not perfectly recover original!"

# Bring the raw shape (uninflate)
dataset_sq   = dataset_gsdr_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)

dataset_sq = np.squeeze(dataset_sq, axis = (2,5)) # (N,T,H,W,F)
print(f'Raw shape of GSDR2d: {dataset_sq.shape}')

# Split back into train/val/test normalized sets ---
# N_train = train_data.shape[0]
# N_val   = val_data.shape[0]
# train_norm = dataset_sq[:N_train]
# val_norm   = dataset_sq[N_train:N_train + N_val]
# test_norm  = dataset_sq[N_train + N_val:]
test_norm = dataset_sq

# del train_data, val_data, test_data
del test_data

# Gather filenames and derive chunk sizes per file
def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_gsdr2d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each MHD file holds one or more sims along axis=0 of magnetic_field
            n = h5f['t0_fields/A'].shape[0]
        chunks.append(n)
    return files, chunks

# train_files, train_chunks = get_files_and_chunks('train')
# val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    # ('train', train_norm, train_files, train_chunks),
    # ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:

    out_dir = os.path.join(savepath_norm_data_gsdr2d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, H, W, F)
        ptr += sz

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            g = f_out.create_group('t0_fields')
            act_A = chunk[..., 0] # (sz, T, H, W)
            act_B = chunk[..., 1] # (sz, T, H, W)
            g.create_dataset('A',data=act_A,compression='lzf')
            g.create_dataset('B',data=act_B,compression='lzf')

        print(f"[GSDR2D] Saved file: {fname}, chunks={sz}, "
              f"shape(s) A={act_A.shape}, B={act_B.shape}")

print("Normalized GSDR data saved under:", savepath_norm_data_gsdr2d)

