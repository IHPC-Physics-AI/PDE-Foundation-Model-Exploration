# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import os



import math
import logging
from torch.nn.modules.container import Sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# REG = float(os.environ.get("REG", "1"))
ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    """
    def __init__(self, width = 32, num_blocks=8, channel_first = False,sparsity_threshold=0.01, modes = 32,hard_thresholding_fraction=1, hidden_size_factor=1, act='gelu'):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"



        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # x = torch.fft.rfft2(x, dim=(1, 2))

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        ## for ab study
        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")



        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x




_logger = logging.getLogger(__name__)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATION[act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, mixing_type = 'afno', double_skip = True, width = 32, n_blocks = 4, mlp_ratio=1., channel_first = True, modes = 32, drop=0., drop_path=0., act='gelu', h=14, w=8,):
        super().__init__()
        # self.norm1 = norm_layer(width)
        # self.norm1 = torch.nn.LayerNorm([width])
        self.norm1 = torch.nn.GroupNorm(8, width)
        # self.norm1 = torch.nn.InstanceNorm2d(width,affine=True,track_running_stats=False)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO2D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first = channel_first, modes = modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act)

        self.norm2 = torch.nn.GroupNorm(8, width)



        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)


        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residual

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128,act='gelu'):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C
    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x










class DPOTNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type = 'afno',in_channels = 1, out_channels = 4, in_timesteps = 1, out_timesteps = 1, n_blocks = 4, embed_dim = 768, out_layer_dim = 32, depth = 12, modes = 32,
                 mlp_ratio=1., n_cls = 12, normalize=False, act='gelu', time_agg='exp_mlp'):
        '''

        :param img_size: input resolution
        :param patch_size: patch size
        :param mixing_type: type of the mixer
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param in_timesteps: number of input timesteps
        :param out_timesteps: number of output timesteps
        :param n_blocks: number of heads/blocks
        :param embed_dim: latent embedding dimension
        :param out_layer_dim: dimension of output convolutional layer
        :param depth: number of layers
        :param modes: number of Fourier modes
        :param mlp_ratio: ratio of MLP dim
        :param n_cls: number of datasets (no influence)
        :param normalize: whether normalize data
        :param act: activation type
        :param time_agg: type of temporal agg layer
        '''
        super(DPOTNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels + 3, embed_dim=out_channels * patch_size + 3, out_dim=embed_dim,act=act)
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls


        h = img_size // patch_size
        w = h // 2 + 1




        self.blocks = nn.ModuleList([
            Block(mixing_type=mixing_type,modes=modes,
                  width=embed_dim, mlp_ratio=mlp_ratio, channel_first = True, n_blocks=n_blocks,double_skip=False, h=h, w=w,act = act)
            for i in range(depth)])


        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * in_channels, embed_dim)


        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)


        ### attempt load balancing for high resolution
        # self.out_layer = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
        #     self.act,
        #     nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
        #     self.act,
        #     nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        # )

        # pseudo-inverse
        self.out_deconv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size)
        self.out_act1 = self.act
        self.out_conv1 = nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1)
        self.out_act2 = self.act
        self.out_conv2 = nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)



        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = mixing_type


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
                # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid


    ### in/out: B, X, Y, T, C
    def forward(self, x):
        inputs0 = x[:, :, :, -1, 0:2]
        inputs0 = inputs0.permute(0, 3, 1, 2)   # B, 2, X, Y
        B, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   #-> B, C, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 3, 1, 2)


        grid = self.get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3, M: (1,128,128,10,4) -> (1,128,128,10,7), L: Same as S/M
        x = rearrange(x, 'b x y t c -> (b t) c x y') #M: (1,128,128,10,7) -> (10,7,128,128), L: Same as S/M
        x = self.patch_embed(x) #M: (10,7,128,128) -> (10,1024,16,16), L: (10,7,128,128) -> (10,1536,16,16)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T) #M: (10,1024,16,16) -> (1,16,16,10,1024), L: (10,1536,16,16) -> (1,16,16,10,1536)

        x = self.time_agg_layer(x) #M: (1,16,16,10,1024) -> (1,16,16,1024), L: (1,16,16,10,1536) -> (1,16,16,1536)

        x = rearrange(x, 'b x y c -> b c x y') #M: (1,16,16,1024) -> (1,1024,16,16), L: (1,16,16,1536) -> (1,1536,16,16)

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        for blk in self.blocks:
            x = blk(x)



        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        # x = self.out_layer(x).permute(0, 2, 3, 1)

        # pseudo-inverse
        x = self.out_deconv(x) #S/M: (1,1024,16,16), L: (1,1536,16,16)

        x = self.out_act1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_conv1(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x = self.out_act2(x) #S/M: (1,32,128,128), L: (1,128,128,128)

        x, ssr = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(x, inputs0)
        # x = self.out_conv2(x) #S/M: (1,32,128,128) -> (1,4,128,128), L: (1,128,128,128) -> (1,4,128,128)

        # x, ssr = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(x, inputs0)

        ones = torch.ones(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype) # noise becomes all ones here
        x = torch.cat([x, ones], dim=1)

        x = x.permute(0, 2, 3, 1)

        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous() # (1,128,128,4) -> (1,128,128,1,4)

        if self.normalize:
            x = x * sigma  + mu

        return x, cls_pred

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr



def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict



def CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(inputs, inputs0):

    inputs_conv2d = inputs

    n_kernel_size = 5  #Parameter, values 3/5/7
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    # inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    # inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    # inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    # inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    x_l, x_u, y_l, y_u = -1, 1, -1, 1
    # x_l, x_u, y_l, y_u = 0, 1, 0, 1
    ext = [x_l, x_u, y_l, y_u]
    t_l, t_u = 0, 5

    # dx, dy, dt
    n_x, n_y, n_t = 128, 128, 101
    # dx = (x_u - x_l) / (n_x - 1) #/ 2
    # dy = (y_u - y_l) / (n_y - 1) #/ 2
    # dt = (t_u - t_l) / (n_t - 1) #/ 2
    dx = dy = 0.015625
    dt = 0.05

    n_padding = int((inputs0.shape[2] - n_x)/2)

    x = np.linspace(x_l, x_u, n_x)
    y = np.linspace(y_l, y_u, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # add BC indicator
    _boundary_left = (x_grid == x_l) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_bottom = (y_grid == y_l) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_right = (x_grid == x_u) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_top = (y_grid == y_u) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_left_bottom = (x_grid == x_l) & (y_grid == y_l)
    _boundary_right_bottom = (x_grid == x_u) & (y_grid == y_l)
    _boundary_right_top = (x_grid == x_u) & (y_grid == y_u)
    _boundary_left_top = (x_grid == x_l) & (y_grid == y_u)

    x_ind_bc = np.zeros((n_y, n_x))
    x_ind_bc[_boundary_left] = 1.
    x_ind_bc[_boundary_bottom] = 2.
    x_ind_bc[_boundary_right] = 3.
    x_ind_bc[_boundary_top] = 4.
    x_ind_bc[_boundary_left_bottom] = 5.
    x_ind_bc[_boundary_right_bottom] = 6.
    x_ind_bc[_boundary_right_top] = 7.
    x_ind_bc[_boundary_left_top] = 8.

    device = inputs.device

    # pad BC indicator
    x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = np.expand_dims(x_ind_bc, axis=0)
    # x_ind_bc = torch.tensor(x_ind_bc)
    # ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], axis=0)
    ind_bc = torch.tensor(ind_bc).to(device)

    # inputs_np = inputs.detach().cpu().numpy()

    # M_u = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    # ni = 0
    # for _t in np.arange(inputs_conv2d.shape[0]):
    #     for _x in np.arange(inputs_conv2d.shape[2]):
    #         for _y in np.arange(inputs_conv2d.shape[3]):
    #             M_u_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
    #             M_u_part = M_u_part.reshape(M_u_part.shape[0], M_u_part.shape[1], -1)
    #             M_u_part = M_u_part.reshape(M_u_part.shape[0], -1)
    #             M_u[ni:ni+1, :] = M_u_part
    #             ni = ni + 1

    # M_v = np.hstack([np.zeros_like(M_u), M_u])  # add the u part
    # M_u = np.hstack([M_u, np.zeros_like(M_u)])  # add the v part
    # M_u, M_v = torch.tensor(M_u).to(device), torch.tensor(M_v).to(device)

    # F.unfold does the same as the triple loop above, but entirely on GPU, which speeds up inference
    # inputs_conv2d: (B, C, H, W), already padded via inputs_conv2d (unpadded)
    patches = F.unfold(inputs_conv2d, kernel_size=n_kernel_size, padding=n_p)
    # patches shape: (B, C*k*k, H*W)
    B_u, CKK, HW = patches.shape
    M_u_raw = patches.permute(0, 2, 1).reshape(B_u * HW, CKK)  # (B*H*W, C*k*k)

    zeros = torch.zeros_like(M_u_raw)
    M_u = torch.cat([M_u_raw, zeros], dim=1).double()   # (B*H*W, 2*C*k*k)
    M_v = torch.cat([zeros, M_u_raw], dim=1).double()   # (B*H*W, 2*C*k*k)

    # get the inputs after flatten
    M_uv_t0 = inputs0
    M_u_t0, M_v_t0 = M_uv_t0[:, 0, :, :].reshape(M_uv_t0.shape[0], -1), M_uv_t0[:, 1, :, :].reshape(M_uv_t0.shape[0], -1)
    M_u_t0, M_v_t0 = M_u_t0.flatten(), M_v_t0.flatten()
    # M_u_t0, M_v_t0 = M_u_t0.to(device), M_v_t0.to(device)
    M_u_t0, M_v_t0 = M_u_t0.to(device).double(), M_v_t0.to(device).double()

    # first iteration

    # Tunable parameters (regularization term)

    # Values below give optimal results for inference of sgedulion/sgmerlion/sglion datasets
    # Adjust according to kernel size
    # reg = REG  # Used for grid_search
    # reg = 1e-1  # kernel 3
    reg = 1  # kernel 5
    # reg = 75 # kernel 7

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3
    u0_0 = ut0.reshape(-1, 1)
    pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC
    pde_M_u_normal_re = ut0/dt - k
    pde_M_v_normal_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_normal_re = vt0/dt

    ### ------left boundary------ ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uW = uC, vW = vC)
    u0_1 = ut0.reshape(-1, 1)
    pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC
    pde_M_u_left_re = ut0/dt - k
    pde_M_v_left_le = vC/dt - Dv*(vE - vC)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_left_re = vt0/dt

    ### ------bottom boundary------ ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    # uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uS = uC, vS = vC)
    u0_2 = ut0.reshape(-1, 1)
    pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC
    pde_M_u_bottom_re = ut0/dt - k
    pde_M_v_bottom_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - vC)/(dy*dy) - uC + vC
    pde_M_v_bottom_re = vt0/dt

    ### ------right boundary------ ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uE = uC, vE = vC)
    u0_3 = ut0.reshape(-1, 1)
    pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC
    pde_M_u_right_re = ut0/dt - k
    pde_M_v_right_le = vC/dt - Dv*(-vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_right_re = vt0/dt

    ### ------top boundary------ ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    # uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uN = uC, vN = vC)
    u0_4 = ut0.reshape(-1, 1)
    pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC
    pde_M_u_top_re = ut0/dt - k
    pde_M_v_top_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(-vC + vS)/(dy*dy) - uC + vC
    pde_M_v_top_re = vt0/dt


    pde_M_u_le = torch.cat([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le], axis=0)
    pde_M_v_le = torch.cat([pde_M_v_normal_le, pde_M_v_left_le, pde_M_v_bottom_le, pde_M_v_right_le, pde_M_v_top_le], axis=0)
    pde_M_u_re = torch.cat([pde_M_u_normal_re.reshape(-1, 1), pde_M_u_left_re.reshape(-1, 1), pde_M_u_bottom_re.reshape(-1, 1), pde_M_u_right_re.reshape(-1, 1), pde_M_u_top_re.reshape(-1, 1)], axis=0)
    pde_M_v_re = torch.cat([pde_M_v_normal_re.reshape(-1, 1), pde_M_v_left_re.reshape(-1, 1), pde_M_v_bottom_re.reshape(-1, 1), pde_M_v_right_re.reshape(-1, 1), pde_M_v_top_re.reshape(-1, 1)], axis=0)
    pde_M_le = torch.cat([pde_M_u_le, pde_M_v_le], axis=0)
    pde_M_re = torch.cat([pde_M_u_re, pde_M_v_re], axis=0)
    pde_M_re = pde_M_re.double()  # change to 64 bits

    pde_M_le = pde_M_le.double()

    # start = time.time()

    # GPU version 1
    # kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1], dtype=torch.float64).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

    # GPU version 2, slightly faster compared to version 1
    A = reg * torch.eye(pde_M_le.shape[1], dtype=torch.float64).to(device) + (pde_M_le.T @ pde_M_le)
    kernel_parameters = torch.linalg.solve(A, pde_M_le.T @ pde_M_re)

    # CPU version, significantly slower than GPU
    # pde_M_le1 = pde_M_le.detach().cpu().numpy()
    # pde_M_re1 = pde_M_re.detach().cpu().numpy()
    # kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le1.shape[1]) + (pde_M_le1.T @ pde_M_le1)) @ pde_M_le1.T @ pde_M_re1
    # kernel_parameters = torch.tensor(kernel_parameters).to(device)


    # end = time.time()
    # runtime = end - start

    # iteration to calculate the nonlinear term
    # Number of nonlinear terms can be changed
    # Values 3/4/5, to test for optimal results
    for i in np.arange(3):
        ### ------normal PDE------ ###  0
        _C = torch.eq(ind_bc, 0)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_0 = uC @ kernel_parameters

        # compute PDE matrix
        pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC

        ### ------left boundary------ ###  1
        _C = torch.eq(ind_bc, 1)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_1 = uC @ kernel_parameters

        # compute PDE matrix (uW = uC, vW = vC)
        pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC

        ### ------bottom boundary------ ###  2
        _C = torch.eq(ind_bc, 2)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        u0_2 = uC @ kernel_parameters

        # compute PDE matrix (uS = uC, vS = vC)
        pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC

        ### ------right boundary------ ###  3
        _C = torch.eq(ind_bc, 3)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_3 = uC @ kernel_parameters

        # compute PDE matrix (uE = uC, vE = vC)
        pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC

        ### ------top boundary------ ###  4
        _C = torch.eq(ind_bc, 4)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uS = M_u[_S, :]
        u0_4 = uC @ kernel_parameters

        # compute PDE matrix (uN = uC, vN = vC)
        pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC

        pde_M_u_le = torch.cat([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le], axis=0)
        pde_M_le = torch.cat([pde_M_u_le, pde_M_v_le], axis=0)
        pde_M_le = pde_M_le.double()
        # pde_M_re = torch.cat([pde_M_u_re, pde_M_v_re], axis=0)
        # pde_M_re = pde_M_re.double()  # change to 64 bits

        # GPU version 1
        # kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1], dtype=torch.float64).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

        # GPU version 2, faster than version 1
        A = reg * torch.eye(pde_M_le.shape[1], dtype=torch.float64).to(device) + (pde_M_le.T @ pde_M_le)
        kernel_parameters = torch.linalg.solve(A, pde_M_le.T @ pde_M_re)

        # CPU version, significantly slower than GPU
        # pde_M_le1 = pde_M_le.detach().cpu().numpy()
        # kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le1.shape[1]) + (pde_M_le1.T @ pde_M_le1)) @ pde_M_le1.T @ pde_M_re1
        # kernel_parameters = torch.tensor(kernel_parameters).to(device)

        # test error
        # _C = torch.eq(ind_bc, 0)
        # _C = _C.reshape(_C.shape[0], -1)
        # _C = _C.flatten()
        # uC = M_u[_C, :]
        # u0_0_new = uC @ kernel_parameters
        # error = torch.mean((u0_0_new - u0_0)**2)
        # ttt = 1

    # pseudo inverse prediction results
    u, v = M_u @ kernel_parameters, M_v @ kernel_parameters
    u, v = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1)
    u, v = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v = torch.unsqueeze(u, axis=1), torch.unsqueeze(v, axis=1)
    outputs_pseudo_inverse = torch.concat([u, v], axis=1)
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)


    return outputs_pseudo_inverse, ssr




if __name__ == "__main__":
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(4, 20, 20, 6, 3)
    net = DPOTNet(img_size=20, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32,normalize=True)
    y,_ = net(x)
    print(y.shape)