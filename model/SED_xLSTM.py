import torch
import torch.nn as nn
from functools import partial
import numpy as np
from torch import Tensor
from einops.layers.torch import Reduce
from model.xlstm import xLSTM as xlstm


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, norm_layer=None):
        super().__init__()
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(self.in_c, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, num_patches]
        # transpose: [B, C, num_patches] -> [B, num_patches, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x




class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.xlstm = xlstm(dim, 128, 4, ['m', 'm'], batch_first=True, proj_factor_slstm=1, proj_factor_mlstm=2)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x,state = self.xlstm(x)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class xLSTMEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.2, attn_drop_ratio=0.2, drop_path_ratio=0.2, norm_layer=None, act_layer=None):

        super(xLSTMEncoder, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_c = in_c
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.01)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class SpatialBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, act_layer, norm_layer, kernel_size_spa, drop_ratio=0.2, drop_path_ratio=0.2):
        super(SpatialBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.spa_attn = SpatialAttention(kernel_size_spa=kernel_size_spa)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.spa_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size_spa):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size_spa, stride=(1, 1), padding='same')
        self.tanh = nn.Tanh()

    def forward(self, x):
        source = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.tanh(x)
        x = x * source
        x = x + source
        return x


class Spatial(nn.Module):
    def __init__(self, in_c_spa, depth_spa, embed_dim, kernel_size_spa):
        super(Spatial, self).__init__()
        self.phase = nn.Conv2d(in_channels=2, out_channels=in_c_spa, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=in_c_spa)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(*[
            SpatialBlock(dim=embed_dim, mlp_ratio=4, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                         kernel_size_spa=kernel_size_spa)
            for _ in range(depth_spa)])
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(embed_dim)
        

    def forward(self, x):
        x = self.relu(self.norm(self.phase(x)))
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class xlstmDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, lstm_type):
        super(xlstmDecoder, self).__init__()
        self.xlstm = xlstm(embed_dim, 128, num_heads, lstm_type, batch_first=True, 
                           proj_factor_slstm=1, proj_factor_mlstm=2)
        
        self.conv1 = nn.Conv1d(12, 48, kernel_size=3, stride=1, padding='same')
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.conv2 = nn.Conv1d(48, 12, kernel_size=3, stride=1, padding='same')

        
        self.fc1 = nn.Conv1d(12, 12, kernel_size=3, stride=1, padding='same')
        self.fc2 = nn.Conv1d(12, 12, 1, 1)

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(0.2)


    def forward(self, x):
        x = self.norm(x)
        skip = x
        x = self.fc1(x)
        x1 = self.fc2(x)
        x1, state = self.xlstm(x1)
        x1 = self.drop(x1)
        
        x2 = self.silu(x)
        x = x2 * x1 + skip
        x = self.conv2(self.norm(self.act(self.conv1(x)))) + skip
        return x

class Recalibration(nn.Module):
    def __init__(self, img_size):
        super(Recalibration, self).__init__()
        
        self.fre_linear1 = nn.Linear(img_size[1], img_size[1] * 4)
        self.fre_linear2 = nn.Linear(img_size[1] * 4, img_size[1])
        self.fre = nn.Conv1d(12, 12, 3, 1, padding='same')
        self.channel = nn.Conv1d(12, 12, 3, 1, padding='same')

        self.gelu = nn.GELU()
        self.channel_linear1 = nn.Linear(img_size[1], img_size[1] * 4)
        self.channel_linear2 = nn.Linear(img_size[1] * 4, img_size[1])



    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x1 = self.gelu(self.fre_linear2(self.fre_linear1(x1)))
        x1 = x * x1
        
        x2, _ = torch.max(x, dim=1, keepdim=True)
        x2 = self.gelu(self.channel_linear2(self.channel_linear1(x2)))
        x2 = x * x2

        return x1 + x2 + x

class SED_xLSTM(nn.Module):
    def __init__(self, num_classes, in_c, img_size, patch_size, embed_dim, depth, num_heads, in_c_spa,
                 depth_spa, kernel_size_spa):
        super(SED_xLSTM, self).__init__()
        self.spatial_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c_spa, embed_dim=embed_dim)
        self.spatial = Spatial(in_c_spa=in_c_spa, depth_spa=depth_spa, embed_dim=embed_dim,
                               kernel_size_spa=kernel_size_spa)
        self.embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.xlstmencoder = xLSTMEncoder(img_size=img_size, patch_size=patch_size, in_c=in_c, num_heads=num_heads,
                                 depth=depth, embed_dim=embed_dim)
        
        self.xlstmdecoder = nn.Sequential(xlstmDecoder(embed_dim, 4, ['m', 'm']),
                                   xlstmDecoder(embed_dim, 4, ['m', 'm']))
        self.output = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(embed_dim * 12, embed_dim * 4),
                        nn.GELU(),
                        nn.Dropout(0.4),
                        nn.Linear(embed_dim * 4, embed_dim),
                        nn.GELU(),
                        nn.Dropout(0.4),
                        nn.Linear(embed_dim, num_classes))
        self.conv = nn.Conv2d(3, 3, (1, 2), (1, 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.re = Recalibration((12, 256))
 
    def forward(self, x):
        x_spa = self.spatial_embed(self.spatial(x))
        x = self.xlstmencoder(self.embed(x))
        x = self.xlstmdecoder(self.re(x + x_spa))
        x = self.output(x)
        return x



def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = SED_xLSTM(num_classes=args.num_classes,
                          in_c=args.in_c,
                          img_size=(9, 256),
                          patch_size=args.patch_size,
                          embed_dim=args.embed_dim,
                          depth=args.depth,
                          num_heads=args.num_heads,
                          in_c_spa=args.in_c_spa,
                          depth_spa=args.depth_spa,
                          kernel_size_spa=args.kernel_size_spa)
    elif args.dataset_name == 'JFPM':
        model = SED_xLSTM(num_classes=args.num_classes,
                          in_c=args.in_c,
                          img_size=(8, 256),
                          patch_size=args.patch_size,
                          embed_dim=args.embed_dim,
                          depth=args.depth,
                          num_heads=args.num_heads,
                          in_c_spa=args.in_c_spa,
                          depth_spa=args.depth_spa,
                          kernel_size_spa=args.kernel_size_spa)
    else:
        return None
    return model
