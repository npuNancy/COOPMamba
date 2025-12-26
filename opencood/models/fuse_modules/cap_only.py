import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v1', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N=H*W, C) 
        skip = input
        input = self.norm(input)
        output = self.block(input)
        return output + skip

class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N=H*W, C) | input1: (B, N=H*W, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        return output + skip


class caponly(nn.Module):
    def __init__(self, dim, H, W, depth=4):
        super().__init__()
        self.sp1_mamba_layers = nn.ModuleList([])
        self.sp2_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.sp1_mamba_layers.append(SingleMambaBlock(dim, H, W))
            self.sp2_mamba_layers.append(SingleMambaBlock(dim, H, W))
        self.sp1_cross_mamba = CrossMambaBlock(dim, H, W)
        self.sp2_cross_mamba = CrossMambaBlock(dim, H, W)
        self.out_proj = nn.Linear(dim, dim)
    
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self,data,record_len):
        b,c,h,w =data.shape
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(data, record_len)

        out = []
        for xx in split_x:
            xaverage =torch.mean(xx,dim=0,keepdim=True)
            xaverage= rearrange(xaverage, 'b c h w ->b (h w) c', h=h, w=w)
            for sp2_layer in self.sp2_mamba_layers:
                 
                 xaverage=sp2_layer(xaverage)
            xaverage =rearrange(xaverage, 'b (h w) c ->b  c h w', h=h, w=w)
            out.append(xaverage)
        return torch.cat(out, dim=0)    