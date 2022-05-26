import numpy as np
import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, bilinear_sampler_with_mask

import importlib

alt_cuda_corr = importlib.util.find_spec("alt_cuda_corr")
# if alt_cuda_corr is not compiled, then alt_cuda_corr == None

class CorrBlock:
    def __init__(self, fmap1:torch.Tensor, fmap2:torch.Tensor, num_levels:int=4, radius:int=4, normalize_sqrt:bool=True):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.normalize_sqrt = normalize_sqrt

        # all pairs correlation
        corrH1W1H2W2 = self.corr(fmap1, fmap2, self.normalize_sqrt)

        batch, h1, w1, dim, h2, w2 = corrH1W1H2W2.shape
        corrH1W1H2W2 = corrH1W1H2W2.reshape(batch*h1*w1, dim, h2, w2)

        self.shape1 = fmap1.shape
        self.shape2 = fmap2.shape
        
        self.corr_pyramid.append(corrH1W1H2W2)
        for i in range(self.num_levels-1):
            corrH1W1H2W2 = F.avg_pool2d(corrH1W1H2W2, 2, stride=2)
            self.corr_pyramid.append(corrH1W1H2W2)

    def __call__(self, coords:torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)

            # minor bug: output is in yx format, not like final output flow xy 
            delta_yx = torch.stack(torch.meshgrid(dy, dx), dim=-1).to(coords.device, dtype=coords.dtype)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_yx_lvl = delta_yx.view(1, 2*r+1, 2*r+1, 2)
            coords_yx_lvl = centroid_lvl + delta_yx_lvl

            corr_yx = bilinear_sampler(corr, coords_yx_lvl)
            corr_yx = corr_yx.view(batch, h1, w1, -1)
            out_pyramid.append(corr_yx)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().to(dtype=coords.dtype)

    def corr(self, fmap1:torch.Tensor, fmap2:torch.Tensor, normalize_sqrt:bool):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corrH1W1H2W2 = torch.matmul(fmap1.transpose(1,2), fmap2)
        corrH1W1H2W2 = corrH1W1H2W2.view(batch, ht, wd, 1, ht, wd)
        if normalize_sqrt :
            corrH1W1H2W2 = corrH1W1H2W2  / torch.sqrt(torch.tensor(dim).to(dtype=fmap1.dtype))
        return corrH1W1H2W2


class AlternateCorrBlock:
    @torch.jit.unused
    def __init__(self, fmap1:torch.Tensor, fmap2:torch.Tensor, num_levels:int=4, radius:int=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

        if alt_cuda_corr is None:
            print("alt_cuda_corr Module is not found - did you compile it? ")
    
    @torch.jit.unused
    def __call__(self, coords:torch.Tensor) -> torch.Tensor:
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).to(dtype=fmap1.dtype))


