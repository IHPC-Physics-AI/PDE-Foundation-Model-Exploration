import torch
from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data.pbdl_module import PBDLDataModule
from pdetransformer.core.separate_channels import PDETransformer, Supervised
from pdetransformer.data import MultiDataModule

# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
#     strategy = SingleStepSupervised(model)
#
#     raw_dir = '/Volumes/T7/New_Data/2d_gs_rd/data'
#     # raw_dir = '/Users/sky/Git/pde-transformer/datasets'
#     path_config = {'2D_APE_xxl': raw_dir}
#     # {'gs_theta': '/Volumes/T7/APEBench/PDEtransformer'}
#
#     dataset = PBDLDataModule(
#         path_index=path_config,
#         dataset_name='gs_theta',
#         dataset_type='2D_APE_xxl',
#         unrolling_steps=1,
#         test_unrolling_steps=29,
#         batch_size=1,
#         num_workers=0,
#         cache_strategy='testOnly',
#         normalize_data='mean-std',
#         normalize_const=None
#     )
#
#     # In PBDLDataModule
#     dataset.setup(stage='test')
#     test_loader = dataset.test_dataloader()
#
#     test_list = list(test_loader) # gs_theta, data:(10,30,2,256,256), constants_norm/constants:(10,2),
#     data = test_list[0]
#
#     for key in ['PDE', 'Fields', 'Constants']:
#         data['physical_metadata'][key] = data['physical_metadata'][key].long()
#
#     # in SingleStepSupervised
#     prediction, reference = strategy.predict(data, device=device, num_frames=30)
#     print('Done')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='sc-s').to(device)
    strategy = Supervised(model, timesteps=2)

    dataset = 'gs_theta'
    dataset_name = 'gs_theta'

    params_data = {
        'path_index':
            {'2D_APE_xxl': '/Volumes/T7/New_Data/'},
        'dataset_names': [ dataset ],
        'dataset_type': '2D_APE_xxl',
        'unrolling_steps': 1,
        'test_unrolling_steps': 7,
        'batch_size': 1,
        'num_workers': 1,
        'cache_strategy': 'testOnly',
        'different_resolution_strategy': 'none',
        'normalize_data': 'mean-std',
        'normalize_const': 'None',
        'downsample_factor': 1,
        'max_channels': 2,
    }

    data_module = MultiDataModule(**params_data)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    data = next(iter(test_loader))
    prediction, reference = strategy.predict(data, device=device, num_frames=8)

    print('Done')



if __name__ == "__main__":
    main()