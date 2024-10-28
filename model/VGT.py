import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        # x = self.norm(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size * num_layers, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        batch_size = x.shape[1]
        hidden = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(device)
        rr, hn = self.gru(x, hidden)
        hn = hn.transpose(0, 1).reshape(batch_size, 2 * self.num_layers * self.hidden_size)
        return self.dropout(self.linear(hn))


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
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


class Vit_attn(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.2, attn_drop_ratio=0.2,
                 drop_path_ratio=0.2, norm_layer=None, act_layer=None):

        super(Vit_attn, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_c = in_c
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=0.2,
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, embed_spa):
        super(SpatialAttention, self).__init__()
        self.phase = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.tanh = nn.Tanh()
        self.embed = embed_spa

    def forward(self, x):
        x = self.relu(self.bn(self.phase(x)))
        source = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.tanh(x)
        x = x * source
        x = x + source
        return self.embed(x)


class VGT(nn.Module):
    def __init__(self, num_classes, in_c, img_size, patch_size, embed_dim, depth, num_heads, num_layers):
        super(VGT, self).__init__()
        self.spatial_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.spatial = SpatialAttention(embed_spa=self.spatial_embed)
        self.embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.vit_attn = Vit_attn(img_size=img_size, patch_size=patch_size, in_c=in_c, num_heads=num_heads,
                                 depth=depth, embed_dim=embed_dim)
        self.gru = GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers, output_size=num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.vit_attn(x).transpose(0, 1)
        x = self.gru(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = VGT(num_classes=args.num_classes,
                    in_c=args.in_c,
                    img_size=(30, 256),
                    patch_size=args.patch_size,
                    embed_dim=args.embed_dim,
                    depth=args.depth,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers)
    elif args.dataset_name == 'JFPM':
        model = VGT(num_classes=args.num_classes,
                    in_c=args.in_c,
                    img_size=(8, 256),
                    patch_size=args.patch_size,
                    embed_dim=args.embed_dim,
                    depth=args.depth,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers)
    else:
        return None
    return model
